import os
import io
import sys
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from groq import Groq
from dotenv import load_dotenv

# CRITICAL FIX: Restored PyInstaller path logic so the packaged app can find index.html
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(base_path, ".env"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

@app.get("/")
async def get():
    html_path = os.path.join(base_path, "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/interview")
async def websocket_endpoint(
    websocket: WebSocket, 
    role: str = "Operator", 
    jd: str = "Generic Job Specification"
):
    await websocket.accept()
    print(f"🟢 Channel Bound: Target Role [{role}] with dynamic JD integration active.")
    
    # FIXED: Variables are only declared once now
    topics_covered = []
    full_transcript = []

    # Injecting the dynamically extracted JD straight into the AI assessment logic
    system_instruction = f"""
    You are an expert behavioral economist assisting an executive interviewer. 
    The candidate is applying for the role of: {role}.
    
    TARGET JOB REQUIREMENTS (JD):
    {jd}
    
    Analyze the incoming candidate text for core psychological and economic behaviors:
    Look for: Loss Aversion, Sunk Cost Fallacy, Present Bias, Overconfidence, Herd Mentality, Status Quo Bias.
    
    You MUST output your response matching this text structure exactly:
    Transcript: [Output the raw text exactly as provided by the user]
    Trait Detected: [Name of the trait in English]
    Analysis: [Provide a sharp 1-2 sentence economic critique on how this behavioral trait affects their suitability for the duties in the provided JD]
    Follow-Up Question: [Provide exactly one conversational probing follow-up question that directly challenges them on a specific task or requirement listed in the JD]
    """

    while True:
        try:
            # 1. Safely catch whatever the frontend sends (Audio OR Text Commands)
            message = await websocket.receive()

            # ROUTE A: Handle Incoming Audio from the Microphone
            if "bytes" in message:
                audio_bytes = message["bytes"]
                if not audio_bytes or len(audio_bytes) < 2000:
                    continue

                print("🎙️ Processing incoming audio block against JD logic mapping...")
                
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "response.webm"

                # Transcribe the answer chunk cleanly via Whisper
                transcription_object = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    language="en",
                    prompt=f"This is an HR job interview for a {role} position.",
                    temperature=0.0,
                    response_format="json"
                )
                
                transcript_text = transcription_object.text.strip()

                if len(transcript_text) < 6:
                    print("🔇 Insufficient data captured. Skipping analytics handler.")
                    continue

                print(f"📝 Raw Transcript Caught: '{transcript_text}'")

                # Save the text to the session memory bank for the final report
                full_transcript.append(transcript_text)

                # Link current text segment with previous history tracks
                history_str = ", ".join(topics_covered) if topics_covered else "None"
                user_prompt = f"Candidate Answer: '{transcript_text}'. (Prior conversation themes: {history_str})"

                # Execute context evaluation with Llama-3.1
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.2,
                )

                ai_response = chat_completion.choices[0].message.content
                print("🤖 BE and JD Alignment Evaluation Complete.")

                # Append only summary strings for history continuity
                topics_covered.append(transcript_text[:40] + "...")
                if len(topics_covered) > 4: 
                    topics_covered.pop(0)

                await websocket.send_text(ai_response)

            # ROUTE B: Handle Text Command to Generate the Final Report
            elif "text" in message:
                command = message["text"]
                if command == "GENERATE_REPORT":
                    print("📊 Compiling final candidate report...")
                    
                    try:
                        all_answers = " ".join(full_transcript)
                        if not all_answers.strip():
                            all_answers = "The candidate did not provide enough transcribed audio during this session."
                        
                        report_instruction = f"""
                        You are an expert Executive HR Director. 
                        Role: {role}
                        JD: {jd}
                        
                        Below is the complete transcript of the candidate's answers:
                        "{all_answers}"
                        
                        Write a final hiring report assessing if they are suitable based entirely on these responses.
                        
                        Format your output PURELY in HTML. Use <h3>, <p>, <ul>, <li>, <blockquote>, and <strong> tags ONLY. Do NOT use markdown code blocks.
                        Include these exact sections in this specific order:
                        <h3>1. Candidate Responses (Transcript)</h3> 
                        (List the candidate's exact statements here using <blockquote> tags so they stand out visually)
                        <h3>2. Candidate Response Summary</h3>
                        <h3>3. JD Alignment Analysis</h3>
                        <h3>4. Behavioral Trait Summary</h3>
                        <h3>5. Final Hiring Verdict (Suitable / Not Suitable)</h3>
                        """
                        
                        report_completion = groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are an HTML generator. Output ONLY raw HTML. No markdown."},
                                {"role": "user", "content": report_instruction}
                            ],
                            model="llama-3.1-8b-instant",
                            temperature=0.3,
                            max_tokens=2500  # High memory limit for long reports
                        )
                        
                        report_html = report_completion.choices[0].message.content
                        report_html = report_html.replace("```html", "").replace("```", "").strip()
                        
                        await websocket.send_text(f"FINAL_REPORT:{report_html}")
                        
                    except Exception as report_err:
                        print(f"⚠️ Report Generation Error: {report_err}")
                        await websocket.send_text(f"FINAL_REPORT:<p style='color:red;'>⚠️ Error: {report_err}</p>")

        except WebSocketDisconnect:
            print("🔴 Channel disconnected safely.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            break

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting BE Interview Copilot...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    
