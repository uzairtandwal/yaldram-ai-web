import json
import chromadb
import google.generativeai as genai
from config import GEMINI_API_KEY, CHROMA_PATH, TOP_K
from custom_embed import CustomGeminiEmbeddingFunction

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash', generation_config={"response_mime_type": "application/json"})

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
gemini_ef = CustomGeminiEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="youtube_qa", 
    embedding_function=gemini_ef
)

def ask_question(question: str) -> dict:
    fallback_links_str = ""
    try:
        if collection.count() == 0:
            return {
                "answer": "Mujhe aap ke is sawal ka jawab channel ki mojooda videos mein nahi mila (Database khali hai).",
                "sources": []
            }

        results = collection.query(
            query_texts=[question], 
            n_results=3
        )
        
        context_parts = []
        raw_links = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                
                link = f"{meta['url']}&t={int(meta['start_time'])}s"
                title = meta['title']
                raw_links.append(f"- {title}\n  {link}")
                
                context_parts.append(
                    f"Video Title: {title}\n"
                    f"URL: {link}\n"
                    f"Transcript Chunk:\n{doc}\n---"
                )
        
        context_str = "\n".join(context_parts)
        fallback_links_str = "\n\n".join(raw_links)
        
        if not context_str.strip():
            return {
                "answer": "Mujhe is sawal se related koi video nahi mili.",
                "sources": []
            }

        system_prompt = \"\"\"You are an Islamic Q&A Chatbot. You must answer questions based ONLY on the provided Context (transcripts from my YouTube channel).

RULES:
RULE A: Always write your response entirely in proper Urdu script. Do not use English or Roman Urdu.
RULE B: Look at the Context chunks. Formulate 2 or 3 interesting questions in Urdu related to the topics discussed in these chunks.
RULE C: Directly answer or address these formulated questions using the context, and IMMEDIATELY place the exact YouTube URL directly below each point.
RULE D: Even if the user's specific query is not directly answered in the text, you MUST STILL provide related Urdu questions based on the context and give their video links. DO NOT say "Mujhe jawab nahi mila". Always output related Urdu questions!

Respond in JSON format:
{
    "answer": "The formatted text here",
    "sources": []
}
\"\"\"

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        response = model.generate_content(system_prompt + "\n\n" + user_prompt, safety_settings=safety_settings)
        response_data = json.loads(response.text)
        response_data['sources'] = []

        return response_data

    except Exception as e:
        error_msg = fallback_links_str if fallback_links_str else "Koi video match nahi hui."
        return {
            "answer": f"اس وقت گوگل اے آئی جواب دینے سے قاصر ہے، لیکن مجھے آپ کے سوال سے ملتی جلتی یہ ویڈیوز ملی ہیں:\n\n{error_msg}",
            "sources": []
        }