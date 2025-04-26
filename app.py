import os
import time
import gradio as gr
import requests
import inspect
import pandas as pd
from smolagents import CodeAgent, OpenAIServerModel, Tool#, DuckDuckGoSearchTool, WikipediaSearchTool #, HfApiModel
from requests.adapters import HTTPAdapter, Retry



# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
import os
from google import genai

class YouTubeTranscriptTool(Tool):
    name = "youtube_transcript"
    description = (
        "Fetches the transcript of a YouTube video given its URL or ID.\n"
        "Returns plain text (no timestamps) or raw with timestamps."
    )
    inputs = {
        "video_url": {"type": "string", "description": "YouTube URL or video ID."},
        "raw": {"type": "boolean", "description": "Include timestamps?", "nullable": True}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        api_key = os.environ.get("token")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)

    def forward(self, video_url: str, raw: bool = False) -> str:
        # Accept both full URLs and video IDs
        if not video_url.startswith("http"):
            video_url = f"https://www.youtube.com/watch?v={video_url}"


        prompt = """Analyze the following YouTube video content. Provide a concise summary covering:

                    1.  **Main Thesis/Claim:** What is the central point the creator is making?
                    2.  **Key Topics:** List the main subjects discussed, referencing specific examples, details or technologies mentioned (e.g., AI models, programming languages, projects).
                    3.  **Call to Action:** Identify any explicit requests made to the viewer.
                    4.  **Summary:** Provide a concise summary of the video content.
                    
                    Use the provided title, chapter timestamps/descriptions, and description text for your analysis."""

        try:
            response = self.client.models.generate_content(
                model= "gemini-2.5-pro-exp-03-25",
                contents=[
                    genai.types.Part(text=prompt),
                    genai.types.Part(
                        file_data=genai.types.FileData(file_uri=video_url)
                    )
                ]
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            return f"Error fetching transcript: {str(e)}"


class DuckDuckGoSearchTool(Tool):
    name = "duckduckgo_search"
    description = (
        "Performs a DuckDuckGo web search and returns the top results as plain text."
    )
    inputs = {
        "query": {"type": "string", "description": "Search query."},
        "max_results": {"type": "integer", "description": "Number of results to return.", "nullable": True}
    }
    output_type = "string"

    def __init__(self, timeout=10, max_retries=3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.base_urls = [
            "https://html.duckduckgo.com/html",
            "https://duckduckgo.com/html"
        ]

    def forward(self, query: str, max_results: int = 5) -> str:
        payload = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; GeminiAgent/1.0)",
        }
        for base_url in self.base_urls:
            try:
                resp = self.session.post(
                    base_url,
                    data=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                resp.raise_for_status()
                return self._parse_results(resp.text, max_results)
            except requests.exceptions.Timeout:
                continue  # Try next endpoint
            except requests.exceptions.RequestException as e:
                continue  # Try next endpoint
        return "Error: Unable to fetch DuckDuckGo results after multiple attempts. Please check your VPN or network connection."

    def _parse_results(self, html: str, max_results: int) -> str:
        # Simple HTML parsing for DuckDuckGo results
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for a in soup.select(".result__a")[:max_results]:
            title = a.get_text(strip=True)
            url = a.get("href")
            results.append(f"{title}\n{url}")
        if not results:
            return "No results found."
        return "\n\n".join(results)

   
# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")

        model=OpenAIServerModel(
            model_id="gemini-2.5-flash-preview-04-17", # "gemini-2.5-pro-exp-03-25",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.environ.get("token")
        ) 

        
        self.agent = CodeAgent(
            tools=[
            YouTubeTranscriptTool(),
            #SpeechToTextTool(),
            #ChessEngineTool(),
            DuckDuckGoSearchTool(),  # Built-in web search tool
            #FileReadTool(),          # Custom file reader
            #PDFReaderTool(),         # PDF reader
            #ExcelReaderTool(),       # Excel reader
            #ImageAnalysisTool(),     # Image analysis
                  ], 
            model=model, 
            add_base_tools=True, # Add basic tools like math
            max_steps=10, 
            planning_interval=3
        )

        SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and
        finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated
        list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither use units such as $ or
        percent sign unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the
        digits in plain text unless specified otherwise.
        If you are asked for a comma separated list, apply the above rules depending of whether the element
        to be put in the list is a number or a string.
        """
        self.agent.prompt_templates["system_prompt"] = self.agent.prompt_templates["system_prompt"] + SYSTEM_PROMPT
        
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        
        #fixed_answer = self.agent.run(question) # "This is a default answer."
        
        max_retries = 5  # Retry up to 5 times if quota is exceeded
        interval = 30  # Wait 30 seconds between requests

        for attempt in range(max_retries):
            try:
                fixed_answer = self.agent.run(question)
                print(f"Request succeeded. Waiting {interval} seconds before the next request...")
                time.sleep(interval)  # Wait for the specified interval
                print(f"Agent returning fixed answer: {fixed_answer}")
                return fixed_answer
            except Exception as e:
                msg = str(e)
                if "quota" in msg.lower() or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print(f"Quota exceeded. Attempt {attempt+1}/{max_retries}. Retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                else:
                    return f"An unexpected error occurred: {msg}"

        #print(f"Agent returning fixed answer: {fixed_answer}")
        #return fixed_answer

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)