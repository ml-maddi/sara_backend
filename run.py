"""
Optional script to run the server with ngrok tunnel
Usage: python run_with_ngrok.py
"""
import uvicorn
from pyngrok import ngrok
import nest_asyncio

# Apply nest_asyncio to allow nested event loops (needed for ngrok)
nest_asyncio.apply()

# Set your ngrok auth token
NGROK_AUTH_TOKEN = ""
NGROK_DOMAIN = ""
PORT = 8000

def main():
    # Set ngrok authentication token
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Create ngrok tunnel
    ngrok_tunnel = ngrok.connect(
        PORT,
        host_header='rewrite',
        domain=NGROK_DOMAIN
    )
    
    print('=' * 50)
    print(f'Public URL: {ngrok_tunnel.public_url}')
    print('=' * 50)
    
    # Run the FastAPI app
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False  # Disable reload when using ngrok
    )

if __name__ == "__main__":
    main()