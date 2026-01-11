import io
import asyncio
import soundfile as sf
import torch
import numpy as np
import torchaudio
import librosa
from fastapi import FastAPI, WebSocket
from socketio import ASGIApp, AsyncServer
from fastapi.middleware.cors import CORSMiddleware
import socketio
import base64
import json
from typing import Optional
import logging
from io import BytesIO
import wave
import struct
import tempfile
import os

# Using a convenient VAD iterator wrapper
from silero_vad import VADIterator
# from pysio import AsyncServer

from typing import AsyncGenerator, List, Dict
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Change the function signature to async def
async def generate_response(query: str, history: list, session_id: str = None) -> AsyncGenerator[str, None]:
    try:
        # Step 1: Retrieve relevant documents
        # docs = retriever.get_relevant_documents(query)
        language = detect_language(query)

        if language == "Bangla":
            query = get_in_english(query)
            
        docs = vectorDB.similarity_search(query, k=50)

        print(len(docs))
        context = format_docs(query, docs)

         # Step 2: Format chat history for the prompt
        formatted_history = ""
        if history:
            # Take last 6 messages (3 Q&A pairs) for context
            recent_history = history[-6:] if len(history) > 6 else history
            
            for i in range(0, len(recent_history), 2):
                if i + 1 < len(recent_history):
                    q = recent_history[i]
                    a = recent_history[i + 1]
                    formatted_history += f"Previous Question: {q}\nPrevious Answer: {a}\n\n"

        # Step 2: Format prompt with context and query
        formatted_prompt = custom_prompt.format(
            context=context, 
            question=query,
            history=history,
            language=language
        )

        logger.info(f"formatted_prompt: {formatted_prompt}")
        
        # Step 3: Stream response from LLM
        response_stream = llm.stream(formatted_prompt)  # Use the LLM's stream capability directly

        for token in response_stream:
            logger.debug(f"token: {token}")
            content = token.content
            
            # Your existing cleanup logic is fine
            content = content.replace("\n", "").replace("\\n", "")
            if content:
                yield content
                
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        yield f"Error occurred: {e}"


def detect_language(text):
    bangla_count = sum(0x0980 <= ord(char) <= 0x09FF for char in text)
    english_count = sum(0x0041 <= ord(char) <= 0x007A or 0x0030 <= ord(char) <= 0x0039 for char in text)
    
    if bangla_count > english_count:
        return "Bangla"
    elif english_count > bangla_count:
        return "English"
    else:
        return "Mixed or Unknown"


# Global variables for models
noise_clear_model = None
df_state = None
chat_sessions = {}

async def startup_event():
    """Initialize models on startup"""
    global noise_clear_model, df_state
    logger.info("Starting up - loading models...")
    # Load default model
    noise_clear_model, df_state, _ = init_df()
    logger.info("Models loaded successfully")

# Create FastAPI application
app = FastAPI(title="Audio Processing Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def on_startup():
    """Handle startup events"""
    await startup_event()

# Initialize Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, app)

def asr_output_generate(audioFilePath, lang="bn"):
    """Generate ASR output from audio file"""
    if lang == "bn":
        transcriptions = bn_asr_pipe(audioFilePath)
        logger.info(f"Processing Bangla audio: {audioFilePath}")
    else:
        transcriptions = en_asr_pipe(audioFilePath)
    
    return transcriptions["text"]

async def generate_llm_answer_audio(query: str, session_id: str, lang: str, sid: str, turn_id: str):
    """
    Streams LLM response, buffers it to form 7-word chunks, and sends them
    for TTS and text display.
    """
    text_buffer = ""
    word_buffer = []  # This list will store complete words.
    full_text = ""
    audio_full_chunk = ""
    try:
        # Get chat history for this session
        history = chat_sessions.get(session_id, [])
        # Stream response token by token from the language model
        async for token in generate_response(query, history, session_id):
            # Immediately send the raw token to the frontend for display
            # This ensures the user sees text appearing live
            await sio.emit("llm_answer_chunk", {"text": token, "turn_id": turn_id}, to=sid)

            # Add the new token to our text buffer
            text_buffer += token
            full_text += token

            # Use a while loop in case a token contains multiple words
            while ' ' in text_buffer:
                # Split at the first space to get the complete word and the remainder
                word, text_buffer = text_buffer.split(' ', 1)
                if word:  # Ensure the word is not an empty string
                    word_buffer.append(word)

                # Check if we have a chunk of 7 words ready
                if len(word_buffer) >= 7:
                    # Create the 7-word chunk to process for audio
                    text_to_process = " ".join(word_buffer)
                    logger.info(f"--- Processing 7-word audio chunk: '{text_to_process}' ---")

                    # Generate audio for the 7-word chunk
                    audio_tensor = get_audio_from_text(text_to_process, lang)
                    audio_base64 = tensor_to_base64_mp3(audio_tensor)

                    # Send the audio chunk to the client
                    await sio.emit("audio_chunk", {"audio": audio_base64, "turn_id": turn_id}, to=sid)
                    audio_full_chunk = audio_full_chunk + " <new> " + audio_base64
                    # Clear the word buffer after processing
                    word_buffer = []

        # --- After the loop, process any remaining text ---

        # 1. Add the final part from the text_buffer (which has no trailing space)
        if text_buffer.strip():
            word_buffer.append(text_buffer.strip())

        # 2. Process any remaining words in the word_buffer (could be 1 to 6 words)
        if word_buffer:
            remaining_text = " ".join(word_buffer)
            logger.info(f"--- Processing remaining buffer: '{remaining_text}' ---")

            # Generate audio for the final chunk
            audio_tensor = get_audio_from_text(remaining_text, lang)
            audio_base64 = tensor_to_base64_mp3(audio_tensor)

            # Send the final audio chunk
            await sio.emit("audio_chunk", {"audio": audio_base64, "turn_id": turn_id}, to=sid)
            audio_full_chunk = audio_full_chunk + " <new> " + audio_base64
            
        # Update chat history for this session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        # Add the Q&A pair to history
        chat_sessions[session_id].extend([query, full_text])
        
        # Keep only last 12 messages (6 Q&A pairs) to prevent memory overflow
        if len(chat_sessions[session_id]) > 12:
            chat_sessions[session_id] = chat_sessions[session_id][-12:]
            
        # Signal that the entire process is complete
        await sio.emit("streaming_complete", {
            "answer": full_text,
            "query": query,
            "audio": audio_full_chunk
        }, to=sid)
        logger.info("--- Streaming complete ---")

    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        # Notify the client of the error
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    logger.info(f"Client connected: {sid}")
    await sio.emit('connected', {
        'status': 'Connected to audio processing server',
    }, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")
    # Clean up chat history for this session
    if sid in chat_sessions:
        del chat_sessions[sid]
        logger.info(f"Cleaned up chat history for session: {sid}")

@sio.event
async def audio_float_chunk(sid, data):
    """
    Handles incoming raw float32 audio and sends it directly for transcription.
    """
    try:
        logger.debug(f"Received audio data: {data.keys()}")
        # Decode base64 and convert raw bytes to a float32 numpy array
        audio_np = base64.b64decode(data['audio'])
        
        logger.info(f"Received audio chunk for direct transcription: {len(audio_np)} bytes")
        logger.debug(f"Audio data type: {type(audio_np)}")
        
        # Call the transcription method on the audio processor instance
        transcription = audio_processor.transcribe_audio(audio_np, data['lang'])
        transcription = transcription.replace('â‡', '')
        logger.info(f"Transcription result: '{transcription}'")
        
        # If transcription is successful and not empty, send it to the client
        if transcription and transcription.strip():
            logger.info(f"Direct Transcription: '{transcription}'")
            # The backend emits a 'transcription' event with the text
            turn_id = data.get('turn_id')
            await sio.emit('transcription', {
                'text': transcription,
                'timestamp': data.get('timestamp', 0),
                'turn_id': turn_id
            }, room=sid)

            logger.info(f"Starting streaming for client: {sid}")
            if len(transcription) > 0:
                session_id = sid 
                await generate_llm_answer_audio(transcription, session_id, data['lang'], sid, turn_id)
        else:
            logger.warning("Transcription resulted in empty text, not emitting.")

    except Exception as e:
        logger.error(f"Error processing float audio chunk: {e}", exc_info=True)
        await sio.emit('error', {'message': str(e)}, room=sid)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "vad_loaded": audio_processor.vad_model is not None,
        "transcription_loaded": audio_processor.transcription_model is not None,
        "sample_rate": audio_processor.sample_rate
    }

# Mount the Socket.IO app
app.mount("/", socket_app)