import time, os, chardet, requests, nltk, json, hashlib, struct, base64
import chromadb
import fitz  # PyMuPDF
import ebooklib
import mpld3
import tiktoken

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px

from pathlib import Path
from datetime import datetime
from nltk.tokenize import sent_tokenize
from duckduckgo_search import DDGS
from llama_cpp import Llama, LlamaGrammar
from json2gbnf import generate_grammar_from_schema
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from requests.exceptions import ConnectionError, Timeout, RequestException
from numpy.linalg import norm
from chromadb.utils import embedding_functions
from ast import literal_eval
from ebooklib import epub



##############################################  Functions

def remove_last_stop(string):
    if string.endswith("[stop]"):
        return string[:-len("[stop]")]
    else:
        return "Error: String does not finish with [stop]."

def count_tokens(text):
    return len(text)/3.5
 

def is_valid_json(json_string):
    try:
        json.loads(json_string)
        print(json_string)
        print("Is a Valid JSON")
        return True
    except json.JSONDecodeError:
        print(json_string)
        print("Is NOT a Valid JSON")
        return False

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def scrape_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Exclude navigation and footer by removing these elements
        for nav in soup.find_all('nav'):
            nav.decompose()
        for footer in soup.find_all('footer'):
            footer.decompose()

        # Exclude advertisement and sidebar content
        for ad in soup.find_all(class_=["sidebar", "advertisements"]):  # Replace with actual class names
            ad.decompose()

        # Focus on main content, assuming it's within an 'article' tag or a div with a specific class
        main_content = soup.find('article') or soup.find('div', class_="main-content")  # Replace with actual class name

        # If main content found, extract text from it, else extract from the whole page
        extracted_text = main_content.get_text(strip=True) if main_content else soup.get_text(strip=True)

        return extracted_text[:7500]  # Truncate text to 10000 characters
    except ConnectionError:
        return "Error: Network problem (e.g., DNS failure, refused connection, etc)"
    except Timeout:
        return "Error: Request timed out"
    except RequestException:
        return "Error: There was an ambiguous exception that occurred while handling your request"

################################## Tools for AI
def getwebpage(query):
    global LOGTXT
    LOGTXT = LOGTXT + "\n ********** Attempt to scrape txt from url :  " + str(query) + " **********\n"
    if query.startswith("['") and query.endswith("']"):
        query = query[2:-2]
    
    if is_valid_json(query):
        urls = json.loads(query)
        for url in urls.values():
            print("trying url : ")
            print(url)
            if not is_valid_url(url):
                continue  # Skip invalid URLs
            result = scrape_url(url)
            if result and not result.startswith("Error:"):
                return result  # Return the first successful scrape
        return "Error: No valid URLs or unable to scrape any URLs"
    else:
        cleaned_query = query.replace('"', '')
        
        if is_valid_url(cleaned_query):
            return scrape_url(cleaned_query)
        else:
            return "Error: Invalid URL"
   
def user_input(query):
    result = query
    print("*** Excuting of USER_INPUT action ***")
    #print(result)
    return result

def internet_search(query):
    results = ddgs.text(query, region='wt-wt', safesearch='off', timelimit='y', backend="api", max_results = 5)
    result_list = list(results)
    print("Raw DDGS output")
    print(str(result_list))
    
    #html_string += "</body>\n</html>"
    #result_md = html_to_markdown(html_string)
    result_string = ""  # This string will hold all the concatenated bodies

    for i in range(min(5, len(result_list))):  # This will loop through the first 5 results, or fewer if there are less than 5
        result_string += "Note{}: {}\n".format(i+1, result_list[i]["title"] +"\n"+ result_list[i]['body'] +"\n"+ result_list[i]["href"])
        #result_string += "Note{}: {}\n".format(i+1, result_list[i]["title"] +"|"+ result_list[i]['body'] +"|"+ result_list[i]["href"]) # This adds the body of the current result to the string, with a note number
        #result_string += "Note{}: {}\n".format(i+1, result_list[i]['body'])
    return result_string

def fetch_titles_and_ids(titles):
    global ID_documents_list_for_search
    # Assuming 'document_titles_collection' is already initialized and accessible
    Text_ids_list = []
    #print("titles")
    #print(titles)
    for title in titles:
        # Query the database for each title individually for exact matching
        query_result = document_titles_collection.query(
            query_texts=[title],
            n_results=1  # Assuming each title corresponds to a unique document
        )
        #print("fetching")
        #print(query_result)
        # Extracting the Text ID for the matched document
        if query_result['documents']:
            # Assuming the first document in the result is the exact match
            matched_document = query_result['ids'][0]  # Extract the first element
            print("matched documents")
            print(matched_document)
            
            Text_ids_list.append(matched_document)  # Append the ID string

    #print("Text ID lists")
    #print(Text_ids_list)
        
    ID_documents_list_for_search = flatten_list(Text_ids_list)
    return Text_ids_list

def flatten_list(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            # If the element is a list, extend flat_list with the flattened version of this element
            flat_list.extend(flatten_list(element))
        else:
            # If the element is not a list, append it directly to flat_list
            flat_list.append(element)
    return flat_list

def docu_search(query):
    global ID_documents_list_for_search
    
    # Embedding for the query text
    print("The query is : ")
    print(query)
    result_string = ""
    
    k = 5  # You can adjust this number based on your requirements
    results = collection_Doculibrary.query( 
        query_texts=[query],
        where={"content_hash": {"$in": ID_documents_list_for_search}},  # Modified to use $in with a list of Text IDs
        n_results=k
        )
    print(results)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    for doc, metadata in zip(documents, metadatas):
    # Concatenate the document and the metadata source
        result_string += doc + "\n [Source: " + metadata['source'] + "]\n"
    return result_string

def epub_to_text(file_path):
    book = epub.read_epub(str(file_path))
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters.append(item.get_content())
    
    raw_html = b''.join(chapters)
    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup.get_text()

def pdf_to_text(pdf_path):
    # Open the PDF file
    with fitz.open(pdf_path) as pdf:
        text_content = ""
        # Iterate over each page
        for page_num in range(len(pdf)):
            # Get the page
            page = pdf[page_num]
            # Extract text
            text_content += page.get_text()
    return text_content

def MemorizeResult(query, stored_result_name, user_id, session_id):
    print("STORE RESULT REACHED")

    # Initialize or retrieve your collection library
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-MiniLM-L6-cos-v1")
    collection_library = client.get_or_create_collection(name="My_result_store", embedding_function=sentence_transformer_ef)

    # Check if the document with the same stored_result_name, user_id, and session_id already exists
    existing_document = collection_library.get(
        where={
            "$and": [
                {"stored_result_name": {"$eq": stored_result_name}},
                {"user_id": {"$eq": user_id}},
                {"session_id": {"$eq": session_id}}
            ]
        }
    )

    if not existing_document['documents']:
        # If not, add it to the collection
        metadata = {
            "stored_result_name": stored_result_name,
            "user_id": user_id,
            "session_id": session_id
        }

        # Embed the query and add it to the collection
        embedding = sentence_transformer_ef([query])
        collection_library.add(
            embeddings=embedding,
            documents=[query],
            metadatas=[metadata],
            ids=[f"{user_id}_{session_id}_{stored_result_name}"]  # Generate a unique ID
        )

        state_msg = f"Result {stored_result_name} added to collection and embedded"
    else:
        state_msg = "Document with the same parameters is already existing in the collection"

    print(state_msg)
    return query


def getMemoryResult(query, stored_result_name, user_id, session_id):
    print("GET MEMORY RESULT REACHED")

    # Initialize or retrieve your collection library
    collection_library = client.get_or_create_collection(name="My_result_store")

    # Retrieve the document based on the stored_result_name, user ID, and session ID
    existing_document = collection_library.get(
        where={
            "$and": [
                {"stored_result_name": {"$eq": stored_result_name}},
                {"user_id": {"$eq": user_id}},
                {"session_id": {"$eq": session_id}}
            ]
        }
    )

    if existing_document['documents']:
        print(f"Document with title {stored_result_name} found.")
        return existing_document['documents']
    else:
        print(f"No document found for title {stored_result_name} with given user and session ID.")
        return None

 

def upload_csv(files, csv_delimiter):
    global uploaded_df
    knowledge = ""
    try:
        file_paths = [file.name for file in files]
        #print(file_paths)

        if not file_paths:
            return "Error: No files provided for upload."
        
        path = Path(file_paths[0])
        
        print(path)

        if not os.path.exists(path):
            return f"Error: File at path {path} was not found."

        with open(str(path), 'rb') as file:
            result = chardet.detect(file.read())

            uploaded_df = pd.read_csv(str(path), delimiter=csv_delimiter, encoding=result['encoding'])
            csv_state_msg = str(path) + " loaded"
            dataf = load_df("promptchain/Csv_peek.csv")
            file_name="promptchain/Csv_peek.csv"
            user_tag =""
            assistant_tag=""
            include_history = False
            return csv_state_msg, dataf, user_tag, assistant_tag, include_history, file_name
          
    
    except pd.errors.ParserError:
        csv_state_msg = "Error encountered when parsing sthe CSV file. Check file formatting."
        return csv_state_msg, "nothing"
    except UnicodeDecodeError:
        csv_state_msg = "File encoding not supported. Please ensure file is in the correct format."
        return csv_state_msg, "nothing"
    
    except Exception as e:
        
        return str(e), "nothing"  # General error message if an exception we haven't explicitly handled occurs

    #return basic information : 
    
    return csv_state_msg

def Store_df_from_file(df,path):
    
    df_subset_string = df.iloc[:2].to_string()
    # Generate hash
    tabular_fileID = generate_hash(df_subset_string)
    existing_document = tabular_collection.get(where={"content_hash": tabular_fileID})
    file_title = str(os.path.basename(path))
    if not existing_document['documents']:
        document_titles_collection.add(documents = file_title, ids =tabular_fileID )
        tabular_collection.add(
            documents=df,
            metadatas={"source": file_title, "content_hash": tabular_fileID},
            ids=tabular_fileID
        )
        csv_state_msg ="Document" + file_title + "added to collection and embedded"

    else:

        csv_state_msg = "Document " + file_title + " is already existing in the collection and loaded"

    
    return csv_state_msg, tabular_fileID

def Embed_txt_from_file(files):
    
    global Text_id
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-MiniLM-L6-cos-v1")
    collection_Doculibrary = client.get_or_create_collection(name="my_library", embedding_function=sentence_transformer_ef)
    file_paths = [file.name for file in files]
    file_path = str(Path(file_paths[0]))
    
    file_extension = os.path.splitext(file_path)[1]
    # Detect file extension and handle accordingly
    if file_extension.lower() == '.txt':
        # Load the text file with correct encoding
        with open(str(file_path), 'rb') as file:
            raw_data = file.read()
            encoding = chardet.detect(raw_data)['encoding']
        with open(str(file_path), 'r', encoding=encoding) as file:
            text_content = file.read()
    elif file_extension.lower() == '.epub':
        text_content = epub_to_text(file_path)
    elif file_extension.lower() == '.pdf':
        text_content = pdf_to_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Generate an ID for the text file, could be the name of the file or a hash
    Text_id = generate_hash(text_content)

    # Check if the text file is already in the collection by ID
    existing_document = collection_Doculibrary.get(where={"content_hash": Text_id})
    #print ("Existing Document result :")
    #print(existing_document)
    file_title = str(file_path)
    # If not, embed the text and add it to the collection
    if not existing_document['documents']:
        # add title to title collection
             
        
        max_chars = 1024
        # Make sure text_content is a string or a list of strings here
        chunks_list = slice_text_in_chunk_of_sentences(text_content,max_chars)
        
        #print(file_title)
         # add title to title collection
        document_titles_collection.add(documents = file_title, ids = Text_id)
        #model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
        metadatas_list = [{"source": file_title, "content_hash": Text_id} for _ in chunks_list]
        ids_list = [f"{Text_id}_chunk{index+1}" for index, _ in enumerate(chunks_list)]
        embeddings = sentence_transformer_ef(chunks_list)
        collection_Doculibrary.add(
            embeddings=embeddings,
            documents=chunks_list,
            metadatas=metadatas_list,
            ids=ids_list
        )
        #print (file_path)
        state_msg ="Document" + file_title + "added to collection and embedded"

    else:

        state_msg = "Document " + file_title + " is already existing in the collection and loaded"

    
    return state_msg, Text_id

def generate_hash(text):
    """Generate a unique hash for a given text."""
    return hashlib.md5(text.encode()).hexdigest()

def execute_df_expression(query):
    global uploaded_df
    global Last_image_path, Last_table_path
    global Last_single_expression 
    df = uploaded_df
    # Parse the JSON output
    expression = query      
    # Extract the value associated with the 'df' key
        
    max_attempts = 2
    attempts = 0
    timestamp = datetime.now()
    
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    while attempts < max_attempts:
                
        try:
            if is_valid_json(expression):
                parsed_json = json.loads(expression)
                expression = parsed_json.get('df','')
                Last_single_expression = expression
               
            else:
                expression = expression.strip()
                Last_single_expression = expression
                
            print("UPLOADED DF")
            print(str(df.head(5)))
            print("Tentative to evaluate")
            print(expression)
            result = eval(expression)
            print('Result after eval = ')
            print(result)
            # If the result is a DataFrame or a Series, convert it to HTML
            if isinstance(result, pd.DataFrame):
                headersdf = result.head(5).to_markdown()
                unique_counts = result.nunique().to_markdown()
                num_records = len(result)
                
                #results = result.to_markdown()
                filename = filename + ".csv"
                filename = os.path.join("temp", filename)
                result.to_csv(filename, index=False)
                Last_table_path = filename
                #Comments = "Excecuting the single expression : " +expression+ " produced the following table " + filename + " produced a table with :\n "
                #results = Comments + results
                comments = f"Executing the expression: {expression} saved as {filename} produced the following table:\n"
                #results = result.to_markdown()
                results = comments 
                results += f"\nDataframe 5 first rows\n{headersdf}\n"
                results += f"\n\nNumber of Records: {num_records}\n"
                #results += f"\n\nUnique Values Count Per Column:\n{unique_counts}\n"
                

                return str(results)
            # If the result is a string, return it as is
            elif isinstance(result, pd.Series):
            # Convert Series to DataFrame
                result_df = pd.DataFrame(result)
                
                # Convert DataFrame to Markdown format
                results_markdown = result_df.to_markdown()
                
                # Define filename for saving
                filename = "temp/" + filename + ".csv"
                
                # Ensure 'temp' directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Save DataFrame as CSV
                try:
                    result_df.to_csv(filename, index=False)
                    file_saved_message = "File saved as " + filename
                except Exception as e:
                    file_saved_message = "Error in saving file: " + str(e)
                
                # Prepare final message
                comments = "Executing the expression: " + expression + "\n" + file_saved_message + "\nProduced the following table:\n"
                full_message = comments + results_markdown
                
                # Truncate content if it exceeds 2000 characters
                if len(full_message) > 2000:
                    full_message = full_message[:2000] + "\n[Content truncated due to length]"
                
                return full_message
            
            # If the result is a string, return it as is
            elif isinstance(result, str):
                results = result
                filename = filename + ".txt"
                filename = os.path.join("temp", filename)
                with open(filename, "w") as file:
                # Write the content to the file
                    file.write(results)
                    Last_table_path = filename
                Comments = "Excecuting the single expression : " +expression+ " saved as " + filename + " produced the following text :\n "
                results = Comments + results
                if len(results) > 2000:
                    results = results[:2000] + "\n[Content truncated due to length]"
                 
                return str(results)
            
            elif isinstance(result, plt.Axes):

                fig = result.get_figure()
                print("trying to get result_plt")
                filename = filename + ".png"
                filename = os.path.join("temp", filename)
                fig.savefig(filename)
                print("output created")

                results = "Single Expression : "+ expression +" produced following graphic \n "+ " saved as " + filename
                Last_image_path = filename

                return str(results)
            
            # If the result is of another type, convert it to string
            else:
                results = str(result)
                filename = filename + ".txt"
                filename = os.path.join("temp", filename)
                with open(filename, "w") as file:
                # Write the content to the file
                    file.write(results)
                    Last_table_path = filename
                Comments = "Excecuting the single expression : " +expression+ " saved as " + filename + " produced the following table : \n "
                results = Comments + results
                
                return str(results)
            break  # If evaluation is successful, break the loop
        except Exception as e:
            print(f"Error: {e}")
            # Refine the expression using LLM and try again
            print("Expression attempt is : " + expression)
            context = df.head().to_string()
            full_prompt = "A failed attempt to execute the following pandas python single expression :" + expression + " against the dataframe hereafter "  + context + " produced the following error message : "+ str(e) + " Taking in account the previous error message, please submit hereafter a corrected single expression. Answer is df =  "  
            expression = llm_completion(full_prompt, max_length=128, temperature=0.1, top_p=0.9, grammar='grammar/singleexpression.gbnf')
            attempts += 1

    if attempts == max_attempts:
        results = "Failed to process your request in a valid expression after several attempts."
        filename = os.path.join("temp", filename)
        filename = filename + ".txt"
        with open(filename, "w") as file:
            # Write the content to the file
                file.write(results)
            
        return str(results)
    
def csv_peek(query):
    global uploaded_df
    num_records = len(uploaded_df)
    #data_string = str(first_row_values)
    headers_md = uploaded_df.head(5).to_markdown()
    result =  "The uploaded table contains : " + str(num_records)+ " records.\n " + "Here follows the headers and first 4 rows :\n" +  headers_md
    return result

def json_to_python(json_output):
    data = json.loads(json_output)
    script = ""

    # Function name
    func_name = data.get("function_name")

    # Function parameters
    parameters = data.get("parameters", [])
    params_str = ", ".join([f"{param['name']}: {param['type']}" + (f" = {param['default']}" if "default" in param else "") for param in parameters])

    # Function return type
    return_type = data.get("return_type", "")
    return_annotation = f" -> {return_type}" if return_type else ""

    # Function body
    body = data.get("body", "")

    # Construct the function definition
    script += f"def {func_name}({params_str}){return_annotation}:\n"
    for line in body.splitlines():
        script += f"    {line}\n"

    return script

def execute_python(query):
    # we expect json from python_script.gbnf filter
    print("JSON that embeds pyhon : ****")
    print(query)
    script = json_to_python(query)
    #script = query
    print("Converted python script to be submitted to EXEC : *****")
    print(script)

    try:
        local_vars = {}
        exec(script, globals(), local_vars)

        output_string = ""
        for var in local_vars:
            output_string += f"Variable {var} = {local_vars[var]}\n"
        
        return output_string

    except Exception as e:
        # Return or handle the error message as needed
        return f"An error occurred: {e}"


def List_Library():
    var = document_titles_collection.get()  
    title_entries = var.get('documents', [])
    return title_entries

def slice_text_in_chunk_of_sentences(text, max_chars):

    sentences = sent_tokenize(text)

    # Initialize list to store chunks of sentences
    chunks = []

    # Initialize string to store current chunk
    chunk = ""

    # Iterate through sentences
    for sentence in sentences:
    # If adding the current sentence to the chunk would exceed the maximum number of characters allowed
        if len(chunk) + len(sentence) > max_chars:
        # Add the current chunk to the list of chunks
            chunks.append(chunk)
        # Reset the current chunk
            chunk = ""
    # Add the current sentence to the current chunk
        chunk += sentence + " "
    # If this is the last sentence
        if sentence == sentences[-1]:
        # Add the current chunk to the list of chunks
            chunks.append(chunk)
    return (chunks)

def concatenate_history(user_history, session_id):
    concatenated = []

    # Check if the specific session exists in the user history
    if session_id in user_history:
        # Iterate over the sorted interactions by timestamp within the session
        for timestamp in sorted(user_history[session_id].keys()):
            interaction = user_history[session_id][timestamp]
            concatenated.extend([interaction['user_prompt'], interaction['llm_answer']])

    return ' '.join(concatenated)


def llm_completion(full_prompt, max_length=1024, temperature=0.1, top_p=0.9, grammar=None, **kwargs):
    global MAX_TOK, llm, LOGTXT
    if grammar:
        llama_grammar = LlamaGrammar.from_file(grammar)
    else:
        llama_grammar = None

    print("max length = " + str(max_length))
    print("****** full prompt BEFORE SUBMISSION : " + full_prompt)
    LOGTXT = LOGTXT + "\n********** Engineerd Prompt Before Submission to LLM : ********** \n" + str(full_prompt) +"\n**********\n"
    print("***** COUNT TOKENS :" + str(count_tokens(full_prompt)) + " *********")

    if count_tokens(full_prompt) > MAX_TOK:
        print("Error: The message is too long and exceeds the model's token limit.")
        return "Error: The message is too long and exceeds the model's token limit."

    try:
        llm_answer = llm.create_completion(
            full_prompt, 
            stream=False, 
            repeat_penalty=1.1, 
            max_tokens=max_length, 
            stop =[], 
            echo=False, 
            temperature=temperature, 
            top_p=top_p, 
            mirostat_mode=2, 
            mirostat_tau=4.0, 
            mirostat_eta=1.1,
            grammar=llama_grammar,
            **kwargs
        )
    except ValueError as e:
        print("Error during LLM completion: " + str(e))
        return "Error: Requested tokens exceed context window."

    result = str(llm_answer['choices'][0]['text'])
        
    return result
 
def determine_dataframe(dataf, file_name):

    Last_next_df_value_of_dataf = dataf['next_df'].iloc[-1]
    dataf = load_df(Last_next_df_value_of_dataf)
    file_name = Last_next_df_value_of_dataf

    return dataf, file_name

def concatenate_history(user_history, session_id):
    concatenated_history = ""
    # Check if the specific session exists in the user history
    if session_id in user_history:
        # Sort the session history by timestamp and concatenate
        for timestamp in sorted(user_history[session_id].keys()):
            entry = user_history[session_id][timestamp]
            user_prompt = entry['user_prompt']
            llm_answer = entry['llm_answer']
            concatenated_history += user_prompt + " " + llm_answer + " "
    return concatenated_history

def ResetFullPrompt(query):
    return query[2:-2]

def llm_answer_via_Tool(user_id, session_id, user_prompt, user_tag, assistant_tag, include_history, temperature, repeat_penalty, top_p, max_length, dataf, file_name, context_size):
    global history, next_df_filename
    global MAX_TOK, Last_user_prompt
    global LOGTXT
    LOGTXT =""
    MAX_TOK = context_size - max_length
    print("DATAFRAME")
    print(dataf)
     
    #print(history)
    LOGTXT = LOGTXT + "\n**********\nINITIAL USER PROMPT:\n" + str(user_prompt) +"\n**********\n"
    print("Session ID:", session_id)
    
    # Ensure history is keyed by user_id and session_id
    if user_id not in history:
        history[user_id] = {}
    if session_id not in history[user_id]:
        history[user_id][session_id] = retrieve_conversation_history(user_id, session_id)
         
    # Generate a current timestamp and history entry
    current_time = datetime.now().isoformat()
    history_entry = {'user_prompt': user_prompt, 'llm_answer': "", 'timestamp': current_time}

    # Store the history entry under the session
    history[user_id][session_id][current_time] = history_entry

    max_length = max_length
    temperature = temperature
    next_df_filename = ""

    print('***Values*******')
    print(history[user_id])
    print(session_id)
    print(user_tag)
    print(user_prompt)
    print(len(user_prompt))
    print(assistant_tag)
    # Concatenate history for usage
    if include_history:
        # Pass both user_history and session_id to concatenate_history
        temp_concat = concatenate_history(history[user_id], session_id).strip()
        temp_history = temp_concat[:-len(user_prompt)]
        
        print("TEMP concat")
        print(temp_concat)
        print(len(temp_concat))

        print("TEMP History cut")
        print(temp_history)
        full_prompt = temp_history + " "+ user_tag + user_prompt + assistant_tag
        
    else:
        full_prompt = user_tag + user_prompt + assistant_tag

    print("***** COUNT TOKENS :" + str(count_tokens(full_prompt)) + " *********")

    if count_tokens(full_prompt) > MAX_TOK:
        print("Error: The message is too long and exceeds the model's token limit.")
        return "Error: The message is too long and exceeds the model's token limit."
    
    result = None
    Last_user_prompt = full_prompt
    function_map = {
        'getwebpage': getwebpage,
        'llm_completion': llm_completion,
        'internet_search': internet_search,
        'docu_search': docu_search,
        'MemorizeResult': MemorizeResult,
        'user_input': user_input,
        'csv_peek': csv_peek,
        'execute_df_expression': execute_df_expression,
        'execute_python': execute_python,
        'ReuseDf' : ReuseDf,
        'getMemoryResult' : getMemoryResult,
        'ResetFullPrompt' : ResetFullPrompt
    }

    for index, row in dataf.iterrows():
        action = row['action'].strip()
        function = function_map.get(action)
        if function is None:
            print(f"Function {action} not found.")
            continue

        input_to_function = row['input'].strip().format(full_prompt=full_prompt, result=result)
        option = literal_eval(row['option'].strip()) if row['option'].strip() else {}
        next_df_filename = row['next_df'].strip()
        if action == 'MemorizeResult':
            option['user_id'] = user_id
            option['session_id'] = session_id
        if action == 'getMemoryResult':
            option['user_id'] = user_id
            option['session_id'] = session_id
        

        
        result = str(function(input_to_function, **option))
        if action =='ResetFullPrompt':
            full_prompt = result
        
        print(f"*** Executing sequence steps < {action} > provided following result : ", result)
        LOGTXT = LOGTXT + "\n********** Executing sequence steps < " + str(action) +" > gave following result ********** \n" + str(result) + "\n**********\n"        
    # Update the history entry with the LLM answer
    llm_answer = str(result)
    history[user_id][session_id][current_time]['llm_answer'] = llm_answer

    # Add user and assistant tags
    current_session_history = history[user_id][session_id][current_time]
    current_session_history['user_prompt'] = user_tag + current_session_history['user_prompt']
    current_session_history['llm_answer'] = assistant_tag + current_session_history['llm_answer']

    # Update Memory database
    Memory_database.add(
        documents=[json.dumps(current_session_history)],
        metadatas=[{
            'timestamp': current_time, 
            'session_id': session_id,
            'user_id': user_id  # Include user_id here
        }],
        ids=[current_time]
        )
   
    #test = Memory_database.get(ids=[current_time])
    

    gradio_chat_history = prepare_chat_history_for_gradio(user_id, session_id)
    return gradio_chat_history, LOGTXT,  next_df_filename


def format_for_gradio(history_entry):
    # Converts a single history entry to the format expected by Gradio
    user_prompt = history_entry['user_prompt']
    llm_answer = history_entry['llm_answer']
    return [user_prompt, llm_answer]

def prepare_chat_history_for_gradio(user_id, session_id):
    global history
    if user_id in history and session_id in history[user_id]:
        # Flatten the session history and sort by timestamp
        session_history = history[user_id][session_id]
        sorted_history_entries = sorted(session_history.items(), key=lambda x: x[0])

        # Format each entry in the session history for Gradio
        formatted_history = [format_for_gradio(entry[1]) for entry in sorted_history_entries]
        return formatted_history
    else:
        return []

def retrieve_conversation_history(user_id, session_id):
    # Define the query parameters using $and operator
    query_result = Memory_database.get(
        where={
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"session_id": {"$eq": session_id}}
            ]
        },
        limit=10,
        include=["metadatas", "documents"]
    )

    if query_result:
        # Process and return the result in a format compatible with your history structure
        return format_query_result(query_result)
    else:
        return {}  # Return an empty dictionary if no history is found

def format_query_result(query_result):
    formatted_history = {}

    # Process the 'documents' part of the query result
    for document in query_result.get('documents', []):
        try:
            doc = json.loads(document)  # Assuming each document is a JSON string

            timestamp = doc.get('timestamp', '')
            session_id = doc.get('session_id', '')  # Retrieve the session ID
            user_prompt = doc.get('user_prompt', '')
            llm_answer = doc.get('llm_answer', '')

            if session_id not in formatted_history:
                formatted_history[session_id] = {}

            formatted_history[session_id][timestamp] = {
                'user_prompt': user_prompt,
                'llm_answer': llm_answer,
                'timestamp': timestamp
            }
        except json.JSONDecodeError:
            print(f"Invalid JSON format for document: {document}")
            continue

    return formatted_history

       
def llm_answer_via_Batch(user_id, session_id,user_prompt, user_tag, assistant_tag, include_history, temperature, repeat_penalty, top_p, max_length, dataf, context_size):
    
    global history, chatround_idx, MAX_TOK
        
    # Ensure history is keyed by user_id and session_id
    if user_id not in history:
        history[user_id] = {}
    if session_id not in history[user_id]:
        history[user_id][session_id] = retrieve_conversation_history(user_id, session_id)
         
    # Generate a current timestamp and history entry
    current_time = datetime.now().isoformat()
    history_entry = {'user_prompt': user_prompt, 'llm_answer': "", 'timestamp': current_time}

    # Store the history entry under the session
    history[user_id][session_id][current_time] = history_entry

    MAX_TOK = context_size - max_length
    df = dataf
    max_length = max_length
    temperature = temperature

    # Concatenate history for usage
    if include_history:
        # Pass both user_history and session_id to concatenate_history
        temp_concat = concatenate_history(history[user_id], session_id).strip()
        temp_history = temp_concat[:-len(user_prompt)]
        
        full_prompt = temp_history +" "+ user_tag + user_prompt + assistant_tag
        
    else:
        full_prompt = user_tag + user_prompt + assistant_tag

    print("***** COUNT TOKENS :" + str(count_tokens(full_prompt)) + " *********")

    if count_tokens(full_prompt) > MAX_TOK:
        print("Error: The message is too long and exceeds the model's token limit.")
        return "Error: The message is too long and exceeds the model's token limit."
    
    result = None
    
    # Initialize the action_dict
   
    function_map = {
        'getwebpage': getwebpage,
        'llm_completion': llm_completion,
        'internet_search': internet_search,
        'docu_search': docu_search,
        'MemorizeResult': MemorizeResult,
        'user_input': user_input,
        'csv_peek': csv_peek,
        'execute_df_expression': execute_df_expression,
        'execute_python': execute_python,
        'ReuseDf' : ReuseDf,
        'getMemoryResult':getMemoryResult
        
    }
        # Iterate through each row in the DataFrame and execute actions
    for index, row in dataf.iterrows():
        action = row['action'].strip()
        function = function_map.get(action)
        if function is None:
            print(f"Function {action} not found.")
            continue

        # Prepare input and options for the function
        input_to_function = row['input'].strip().format(full_prompt=full_prompt, result=result)
        option = literal_eval(row['option'].strip()) if row['option'].strip() else {}
        next_df_filename = row['next_df'].strip()
        Next_step_db.update(documents=next_df_filename, ids="unique")

        if action == 'MemorizeResult':
            option['user_id'] = user_id
            option['session_id'] = session_id
        if action == 'getMemoryResult':
            option['user_id'] = user_id
            option['session_id'] = session_id
            
       
        # Call the function and update the result
        result = str(function(input_to_function, **option))
        print(f"*** Now Calling sequence name < {action} > with following result : ", result)

    # Process the final result
    llm_answer = str(result)
    return result   

def enforce_token_limit(history, user_prompt, user_tag, assistant_tag):
    global MAX_TOK
    """
    Ensure the total tokens in history plus the new user prompt 
    doesn't exceed MAX_TOK. If it does, remove the earliest entries.
    """
    total_tokens = count_tokens(concatenate_history(history) + user_tag + user_prompt + assistant_tag)
    print('***Total tokens***')
    print(total_tokens)
    while total_tokens > MAX_TOK and history:
        # Remove the earliest interaction
        removed_interaction = history.pop(0)
        
        total_tokens = count_tokens(concatenate_history(history) + user_tag + user_prompt + assistant_tag)
        
         
    return history

def clear_user_prompt(user_prompt):
    
    user_prompt=""
    return user_prompt

def reset_chat(user_id, session_id):
    global history

    # Generate a unique session ID for the new session
    new_session_id = generate_new_session_id()

    # Optionally, generate or request a summary for the current session
    if user_id in history and session_id in history[user_id]:
        summarize_and_store_session(user_id, session_id)

    # Initialize a new session
    if user_id not in history:
        history[user_id] = {}
    history[user_id][new_session_id] = {}

    # Return an empty chat history or a message indicating a new session
    # Adjust this based on how you want to display it in your Gradio interface
    return [], new_session_id


def generate_new_session_id():
    # Generate a unique session ID
    # Example: "New_" + current timestamp + "_SessionID"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{timestamp}_Session"

def summarize_and_store_session(user_id, session_id):
    global history

    # Step 1: Generate or Retrieve the Summary
    
    session_summary = generate_summary_for_session(history[user_id][session_id])

    # Step 2: Update the History Object
    
    if session_id in history[user_id]:
        history[user_id][session_id]['summary'] = session_summary

    # Step 3: Save to Database
    # Save the updated session data to ChromaDB VectorStore
    Memory_database.add(
        documents=[json.dumps(history[user_id][session_id])],
        metadatas=[{'user_id': user_id, 'session_id': session_id}],
        ids=[session_id]
    )

def generate_summary_for_session(session_history):
    # Logic to generate a summary based on the session history
    # This can be an LLM-based summary generation or any other method
    # For simplicity, let's return a placeholder summary
    return "Placeholder summary of the session."

    
def delete_last_interaction(user_id, session_id):
    global history

    if user_id in history and session_id in history[user_id] and history[user_id][session_id]:
        # Sort the session history by timestamp and remove the last entry
        sorted_session_history = sorted(history[user_id][session_id].items(), key=lambda x: x[0])
        last_entry_timestamp = sorted_session_history[-1][0]
        del history[user_id][session_id][last_entry_timestamp]

        # Update persistent storage if necessary
        # For instance, delete the entry from Memory_database
        # ...

        # Format the remaining entries in the session's history for Gradio
        formatted_history = [format_for_gradio(entry[1]) for entry in sorted_session_history[:-1]]
        return formatted_history
    else:
        # Handle the case where there is no history, no session, or no last interaction
        return False

    


def load_css_styles():
    STYLES = """
    .chuanhu_chatbot {
        background-color: #C7D2E8 !important;
        font-family: system-ui;
        color: White !important;
    }
    .small-big {
    font-size: 9pt !important;
    }
    .panel {
    --radius: 28px;
    --padding: 8px;
    --nested-radius: calc(var(--radius) - var(--padding);
    }   
    .content {
    border-radius: var(--nested-radius);
    }

    .small-small {
    font-size: 7pt !important;
    background: Linen !important;
    padding: 4px !important;
    }
    .small-big-textarea > label > textarea {
    font-size: 11pt !important;
    }
    .highlighted-text {
    background: yellow;
    overflow-wrap: break-word;
    }
    .no-gap {
    gap: 4px !important;
    }
    .group-border {
    padding: 4px;
    border-width: 2px;
    border-radius: 20px;
    border-color: LightGrey;
    border-style: ridge;
    }
    .control-label-font {
    font-size: 8pt !important;
    }
    .control-button {
    background: none !important;
    border-color: Black !important;
    border-width: 2px !important;
    font-size: 11pt !important;
    color: Black !important;
    }
    .icon-button {
    background: none !important;
    border-color: DarkGrey  !important;
    border-width: 2px !important;
    background-color: LightGrey !important;
    color: Linen  !important;
    }
    .center {
    text-align: center;
    }
    .right {
    text-align: right;
    }
    .no-label {
    padding: 5px !important;
    }
    .no-label > label > span {
    display: none;
    }
    .no-label-chatbot {
    border: none !important;
    box-shadow: none !important;
    height: 700px !important;
    }
    .no-label-chatbot > div > div:nth-child(1) {
    display: none;
    }
    .left-margin-30 {
    padding-left: 30px !important;
    }
    .left {
    text-align: left !important;
    }
    .alt-button {
    color: gray !important;
    border-width: 1px !important;
    background: none !important;
    border-color: MintCream !important;
    text-align: justify !important;
    }
    .white-text {
    color: #000 !important;
    }
    """

    small_and_beautiful_theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="rgba(70, 130, 180, 0.1)",
            c100="rgba(70, 130, 180, 0.2)",
            c200="rgba(70, 130, 180, 0.3)",
            c300="rgba(70, 130, 180, 0.4)",
            c400="rgba(70, 130, 180, 0.5)",
            c500="#49475B",
            c600="rgba(70, 130, 180, 0.6)",
            c700="rgba(70, 130, 180, 0.7)",
            c800="rgba(70, 130, 180, 0.8)",
            c900="rgba(70, 130, 180, 0.9)",
            c950="#2E2E3F"
        ),
        secondary_hue=gr.themes.Color(
            c50="#576b95",
            c100="#576b95",
            c200="#576b95",
            c300="#576b95",
            c400="#576b95",
            c500="#576b95",
            c600="#576b95",
            c700="#576b95",
            c800="#576b95",
            c900="#576b95",
            c950="#576b95",
        ),
        neutral_hue=gr.themes.Color(
            name="blue",
            c50="#f9fafb",
            c100="#f3f4f6",
            c200="#e5e7eb",
            c300="#d1d5db",
            c400="#B2B2B2",
            c500="#808080",
            c600="#636363",
            c700="#515151",
            c800="#393939",
            c900="#272727",
            c950="#171717",
        ),
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        button_primary_background_fill="#49475B",               
        button_primary_background_fill_dark="#2E2E3F",          
        button_primary_background_fill_hover="#6C6C7A",         
        button_primary_border_color="#403E4C",                  
        button_primary_border_color_dark="#2B2A37",            
        button_primary_text_color="#ff0000",
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill="#BCBBC7",             
        button_secondary_background_fill_dark="#25252E",        
        button_secondary_text_color="#1E1E28",                  
        button_secondary_text_color_dark="#FFFFFF",
        background_fill_primary="LightGrey",                      
        background_fill_primary_dark="#1C1C24",                 
        block_title_text_color="#49475B",                       
        block_title_background_fill="#b6b2db",                  
        input_background_fill="#e6e6fa",                        
    )

    return STYLES, small_and_beautiful_theme

def load_df(filename):

    with open(filename, 'rb') as file:
        result = chardet.detect(file.read())
    
    # Load DataFrame from CSV file
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(filename, delimiter="|" , encoding=result['encoding'])
        return df
    else:
        return pd.DataFrame("input|action|option|next_df"),"Empty.csv"  # Return empty DataFrame if the file is not a CSV

def load_df_from_df(next_df_filename):
    #print("NEXT DF")
    #print(next_df_filename)
    #print()
    
    with open(next_df_filename, 'rb') as file:
        result = chardet.detect(file.read())
    
    # Load DataFrame from CSV file
    if next_df_filename.lower().endswith('.csv'):
        df = pd.read_csv(next_df_filename, delimiter="|" , encoding=result['encoding'])
        return df,next_df_filename
    else:
        return pd.DataFrame("input|action|option|next_df"),"Empty.csv"  # Return empty DataFrame if the file is not a CSV

def load_df_from_file(files):
    # Detect the file encoding
    file_paths = [file.name for file in files]
    file_name = str(Path(file_paths[0]))
    with open(file_name, 'rb') as file:
        result = chardet.detect(file.read())
    
    # Load DataFrame from CSV file
    if file_name.lower().endswith('.csv'):
        df = pd.read_csv(file_name, delimiter="|" , encoding=result['encoding'])
        Last_next_df_value_of_dataf = df['next_df'].iloc[-1]
        return df,os.path.basename(file_name), Last_next_df_value_of_dataf
    else:
        return pd.DataFrame(),"Empty.csv"  # Return empty DataFrame if the file is not a CSV

def load_dataset_from_file(files):
    # Detect the file encoding
    file_paths = [file.name for file in files]
    file_name = str(Path(file_paths[0]))
    with open(file_name, 'rb') as file:
        result = chardet.detect(file.read())
    
    # Load DataFrame from CSV file
    if file_name.lower().endswith('.csv'):
        df = pd.read_csv(file_name, encoding=result['encoding'])
        
        return df,os.path.basename(file_name)
    else:
        return pd.DataFrame(),"Empty.csv"  # Return empty DataFrame if the file is not a CSV


def load_sequence(filename):
    global sequence_dir
    filename = os.path.join(sequence_dir, filename)
    with open(filename, 'rb') as file:
        result = chardet.detect(file.read())
    
    # Load DataFrame from CSV file
    # return dataf,file_name, next_df_filename
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(filename, delimiter="|" , encoding=result['encoding'])
        Last_next_df_value_of_dataf = df['next_df'].iloc[-1]
        return df,filename, Last_next_df_value_of_dataf
    else:
        return pd.DataFrame(),"Empty.csv"  # Return empty DataFrame if the file is not a CSV
    

def ReuseDf(query):
    global Last_table_path, uploaded_df
    file_name= Last_table_path
    with open(file_name, 'rb') as file:
        result = chardet.detect(file.read())
    
    # Load DataFrame from CSV file
    if file_name.lower().endswith('.csv'):
        uploaded_df = pd.read_csv(file_name, delimiter="," , encoding=result['encoding'])
        csv_state_msg = str(file_name) + " loaded"
        result =""
        return result


def SetNewUploadDf():
    global Last_table_path, uploaded_df
    file_name= Last_table_path
    with open(file_name, 'rb') as file:
        result = chardet.detect(file.read())
    
    # Load DataFrame from CSV file
    if file_name.lower().endswith('.csv'):
        uploaded_df = pd.read_csv(file_name, delimiter="," , encoding=result['encoding'])
        csv_state_msg = str(file_name) + " loaded"
        return csv_state_msg

def delrow(dataf, statement):
    try:
        # Extracting the row index from the string like "[0, 0]"
        index = int(statement.strip("[]").split(",")[0])
        
        # Check if the index is valid, and remove the row at the index
        if 0 <= index < len(dataf):
            dataf = dataf.drop(index).reset_index(drop=True)
        
    except ValueError:
        # Handle the case when the statement is not a valid integer
        print("Invalid index")
    
    return dataf

def move_up(dataf, statement):
    try:
        # Extracting the row index from the statement
        index = int(statement.strip("[]").split(",")[0])
        
        # Check if the index is valid, and move the row up
        if 1 <= index < len(dataf):
            row = dataf.iloc[index]
            dataf.drop(index, inplace=True)
            dataf = pd.concat([dataf.iloc[:index-1], pd.DataFrame(row).T, dataf.iloc[index-1:]]).reset_index(drop=True)
        
    except ValueError:
        # Handle the case when the statement is not a valid integer
        print("Invalid index")
    
    return dataf

def move_down(dataf, statement):
    try:
        # Extracting the row index from the statement
        index = int(statement.strip("[]").split(",")[0])
        
        # Check if the index is valid, and move the row down
        if 0 <= index < len(dataf) - 1:
            row = dataf.iloc[index]
            dataf.drop(index, inplace=True)
            dataf = pd.concat([dataf.iloc[:index+1], pd.DataFrame(row).T, dataf.iloc[index+1:]]).reset_index(drop=True)
        
    except ValueError:
        # Handle the case when the statement is not a valid integer
        print("Invalid index")
    
    return dataf

def save_to_csv(dataf, file_name):
    # Specify the filename and path; here I'm saving it in the current directory
    dataf.to_csv(file_name, sep='|', index=False)
    return f"Saved dataframe to {file_name}"

def save_to_gbnf(gbnf_schema, f_name_gbnf):
    # Specify the filename and path; here I'm saving it in the current directory
    with open(f_name_gbnf, 'w') as file:
        file.write(gbnf_schema)
    
    return f"Saved dataframe to {f_name_gbnf}"

def on_select(evt: gr.SelectData):  # SelectData is a subclass of EventData
    return str(evt.index)

def run_punchcard(user_id, session_id, punchcard_data, user_tag, assistant_tag, include_history, temperature, repeat_penalty, top_p, max_length, dataset_input_data, context_size):
    
    output_list = []

    for index, row in dataset_input_data.iterrows():
        user_prompt = row['full_prompt']
        print(user_prompt)
        
        result = llm_answer_via_Batch(user_id, session_id, user_prompt, user_tag, assistant_tag, include_history, temperature, repeat_penalty, top_p, max_length, punchcard_data, context_size)
    
        formatted_result = f'''    
    <div style="margin-bottom: 20px;">
        <div style="font-weight: bold; font-size: 1.2em; margin-bottom: 0.5em;">{user_prompt}</div>
        <div style="margin-top: 10px;">{result}</div>
    </div>
    '''
        output_list.append(formatted_result)
    
    # Correctly joining and enclosing in HTML tags
    html_output = '<html>' + ''.join(output_list) + '</html>'
    print("*** html batch output ***")
    print(html_output)
    # Return the HTML string
    return html_output
     

    

def update_dropdown(enabled, _):
    if enabled:
        documents_list_string = List_Library()
        return gr.Dropdown(choices=documents_list_string, value=documents_list_string[0], visible=True)
    else:
        # Return an empty list or some default value if checkbox is not checked
        return []

def update_ddown_session(user_id, _):
    Session_dropdown_list_string_init = get_unique_session_ids_for_user(user_id)
    return gr.Dropdown(choices=Session_dropdown_list_string_init, value=Session_dropdown_list_string_init[0], visible=True)

def update_ddown_model_lst( _):
    global Model_dir
    model_lst = scan_gguf_files(Model_dir)
    return gr.Dropdown(choices=model_lst)

def update_ddown_sequence_lst( _):
    global sequence_dir
    sequence_lst = scan_sequence_dir(sequence_dir)
    return gr.Dropdown(choices=sequence_lst)

def toggle_chat_mode(enabled, file_name, user_tag, assistant_tag):
    
    
    if enabled:
        
        dataf = load_df("promptchain/Default_Seq.csv")     
        include_history = True
        file_name = "promptchain/Default_Seq.csv"
    else:
        user_tag =""
        assistant_tag=""
        include_history = False
        dataf = load_df(file_name)

    return user_tag, assistant_tag, dataf, include_history, file_name

def memorize(user_id, session_id):
    global Last_single_expression, Last_image_path, Last_table_path, Last_user_prompt
    
    comment = "description: " + Last_user_prompt
    # if Last_single_expression exist ok otherwise error... 
    
    memory_records = [
        {
            "input": comment,
            "action": "ReuseDf",
            "option": "{}",
            "next_df": "next"
        },
        {
            "input": Last_single_expression,
            "action": "execute_df_expression",
            "option": "{}",
            "next_df": "next"
        }
    ]

    filename = os.path.join("memory", "memory.csv")

    # Try to load existing memories or create a new DataFrame if file not found
    try:
        df_memory = pd.read_csv(filename, delimiter="|")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_memory = pd.DataFrame(columns=memory_records[0].keys())

    # Add new records to the DataFrame
    df_memory = pd.concat([df_memory, pd.DataFrame(memory_records)], ignore_index=True)
    
    # Write the DataFrame to a CSV file
    df_memory.to_csv(filename, sep='|', index=False)
    print("Memorized")

    # Return confirmation or DataFrame based on your needs
    return df_memory

def clear_btn_csv():

    # Define the column names in a list
    column_names = ["input", "action", "option", "next_df"]

    # Create an empty DataFrame with these columns
    csv_memory_df = pd.DataFrame(columns=column_names)
    return csv_memory_df   


def check_output():
    global Last_image_path, Last_table_path
    
    try:
        result_table = None
        #print("LAST TABLE PATH IS ")
        #print(Last_table_path)
        # Check if the last table path is a .csv or .txt file
        if Last_table_path.lower().endswith('.csv'):
            # Detect file encoding for reading
            with open(Last_table_path, 'rb') as file:
                result = chardet.detect(file.read(10000))  # Reading only first 10,000 bytes for efficiency

            # Read CSV into DataFrame and convert to HTML
            df = pd.read_csv(Last_table_path, delimiter=",", encoding=result['encoding'])
            result_table = df.to_html()

        elif Last_table_path.lower().endswith('.txt'):
            # Read text file and convert to HTML
            with open(Last_table_path, 'r', encoding='utf-8') as file:
                text = file.read()
                result_table = f"<html><body><pre>{text}</pre></body></html>"

        else:
            # For other file types, return the path as is
            result_table = Last_table_path

        return Last_image_path, result_table

    except Exception as e:
        # Handle any exceptions that occur
        return f"An error occurred: {e}", None


def retrieve_session(user_id, session_id, limit_number):

    user_records = Memory_database.get(
        where={
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"session_id": {"$eq": session_id}}
            ]
        },
        limit=limit_number,
        include=["metadatas", "documents"]
    )

    formatted_history = []

    if 'documents' in user_records and user_records['documents']:
        for document_json in user_records['documents']:
            try:
                document = json.loads(document_json)
                user_prompt = document.get('user_prompt', '')
                llm_answer = document.get('llm_answer', '')
                formatted_history.append((user_prompt, llm_answer))
            except json.JSONDecodeError:
                print(f"Invalid JSON format for document: {document_json}")
                continue

    return formatted_history


def get_unique_session_ids_for_user(target_user_id):
    unique_session_ids = set()

    # Retrieve records with the specific user_id
    filtered_records = Memory_database.get(where={"user_id": {"$eq": target_user_id}})# Assuming 'metadatas' is a key in the dictionary that contains the relevant data
    metadatas = filtered_records.get('metadatas', [])

    for metadata in metadatas:
        #print("Current metadata:", metadata)  # Debug print

        # Check if 'session_id' is a key in the metadata
        if 'session_id' in metadata:
            unique_session_ids.add(metadata['session_id'])
        else:
            print("session_id not found in metadata")

    return list(unique_session_ids)

def print_whole_database_content():
    # Assuming 'Memory_database' is your database instance
    # Setting a high limit; adjust based on your database size and system capabilities
    result = Next_step_db.get(limit=100)  # Example limit

    # Check and print each component of the result
    for key, value in result.items():
        print(f"--- {key} ---")
        if isinstance(value, list):
            for item in value:
                print(item)
        else:
            print(value)

def print_document_content():
    # Assuming 'Memory_database' is your database instance
    result = Memory_database.get(limit=1000)  # Set a reasonable limit to avoid excessive data retrieval

    if 'documents' in result and result['documents']:
        for document in result['documents']:
            print(document)
    else:
        print("No documents found in the database.")


def scan_gguf_files(directory):
    """Scans the specified directory for files with .gguf extension and returns their names.

    Args:
    directory (str): The path to the directory to scan.

    Returns:
    list: A list of filenames with .gguf extension, or ['empty'] if none found.
    """
    filenames = [file for file in os.listdir(directory) if file.endswith('.gguf')]
    return filenames if filenames else ['empty']

def scan_sequence_dir(directory):
    filenames = [file for file in os.listdir(directory) if file.endswith('.csv')]
    return filenames if filenames else ['empty']


def load_model(model_name, gpu_slice, context_size):
    global Model_dir, llm
    try:
        model_name = Model_dir + "/" + model_name
        llm = Llama(model_path=model_name, n_ctx=context_size, last_n_tokens_size=256, n_threads=4, n_gpu_layers=gpu_slice)
        title_msg = "<h2>GPT Sequencer loaded with : " + str(model_name) + "</h2>"
        return gr.HTML(title_msg, elem_classes=['center'])
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return gr.HTML("<h2>An error occurred while loading the model: {e}</h2>", elem_classes=['center'])

def update_title(model_name):
    title_msg = "<h2><center>GPT Sequencer loaded with : " + str(model_name) + "</center></h2>"
    return gr.Markdown(title_msg, elem_classes=['center'])


################################### initialization

global Text_id
global ID_documents_list_for_search
global Last_image_path, Last_table_path, tokenizer_json_path, MAX_TOK, LOGTXT
global Model_dir

Last_image_path ="temp/1x1.png"
Last_table_path="temp/empty.csv"
Model_dir ="models"
sequence_dir ="promptchain"

ddgs=DDGS()


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-MiniLM-L6-cos-v1")
#sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)

client = chromadb.PersistentClient(path="db/")

Memory_database = client.get_or_create_collection(name="Memory",embedding_function=sentence_transformer_ef )
collection_Doculibrary = client.get_or_create_collection(name="my_library", embedding_function=sentence_transformer_ef)
collection_library = client.get_or_create_collection(name="My_result_store", embedding_function=sentence_transformer_ef)

document_titles_collection = client.get_or_create_collection(name="document_titles", embedding_function=sentence_transformer_ef)
Next_step_db =  client.get_or_create_collection(name="next_steps_sequence", embedding_function=sentence_transformer_ef)
tabular_collection =  client.get_or_create_collection(name="my_tabular", embedding_function=sentence_transformer_ef)

 
if Next_step_db.count() == 0 :
    Next_step_db.add(documents="promptchain/Default_Seq.csv", ids="unique")
    
else:
    Next_step_db.update(documents="promptchain/Default_Seq.csv", ids="unique")
    
history = {}
documents_list_string = List_Library()
ID_documents_list_for_search = []
Session_dropdown_list_string_init=get_unique_session_ids_for_user("Guest")
model_lst = scan_gguf_files(Model_dir)
sequence_lst = scan_sequence_dir(sequence_dir)


############### Gradio interface

STYLES, small_and_beautiful_theme = load_css_styles()

with gr.Blocks(css=STYLES, theme=small_and_beautiful_theme) as demo:
    
    with gr.Tab("Chat Interface", elem_classes=['no-label', 'small-small']):
        
        title_md = gr.HTML("<h2>GPT-Sequencer: empty </h2>", elem_classes=['center'])
        
        with gr.Row():

           
            with gr.Column(scale=3, elem_classes=['group-border']):
                
                with gr.Tab("Mode", elem_classes=['no-label', 'small-small']):
                    with gr.Column(scale=3, elem_classes=['group-border']):
                            Chatbot_Mode_checkbox = gr.Checkbox(label="Chatbot")
                            include_history = gr.Checkbox(label="History", show_label = True, value=True, elem_classes=['small-small', 'icon-button'])
                    with gr.Column(scale=3, elem_classes=['group-border']):
                        file_name = gr.Textbox(label="Active Sequence file", value = "promptchain/Default_Seq.csv", interactive = False)
                        next_df_filename = gr.Textbox(label="Next Sequence file",  interactive = False)
                   # Last_table_path= gr.Textbox(label="Last table")
                    with gr.Column(scale=3, elem_classes=['group-border']):
                        refresh_sequence_lst_btn = gr.Button("Upd. Seq. list", scale=1, elem_classes=['control-button'])
                        sequence_lst_ddown = gr.Dropdown(label="Load Another Seq.", choices=sequence_lst, max_choices=1)
                        load_sequence_btn = gr.Button("Load sequence", scale=1, elem_classes=['control-button'])
                        refresh_sequence_lst_btn.click(fn=update_ddown_sequence_lst,inputs=[sequence_lst_ddown], outputs=[sequence_lst_ddown])
                        
                    
                    reset_btn = gr.Button("🔄 New Session", elem_classes=['control-button'], scale=3)
                    back_btn = gr.Button("Back", elem_classes=['control-button'], scale=3)
                    
                    next_df = gr.Dataframe(visible=False)
                    btn_memorize = gr.Button("Save Interaction")
                    
                
                with gr.Tab("Settings", elem_classes=['no-label', 'small-small']):
                    
                    user_tag = gr.Textbox("[User]: ", elem_classes=['no-label', 'small-small'])
                    assistant_tag = gr.Textbox("[Assistant]: ", elem_classes=['no-label', 'small-small'])
                    temperature = gr.Slider(minimum=0, maximum=1.5, value=0.1, step=0.1, label="Temp.", elem_classes=['no-label', 'small-small'])
                    repeat_penalty = gr.Slider(minimum=0, maximum=2, value=1.1, step=0.1, label="Repeat", elem_classes=['no-label', 'small-small'])
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.01, label="Top_P", elem_classes=['no-label', 'small-small'])
                    max_length = gr.Slider(minimum=1, maximum=1024, value=128, step=1, label="Lenght", elem_classes=['no-label', 'small-small'])

                                    
                
                with gr.Tab("Library", elem_classes=['no-label', 'small-small']):
                    
                    Tools_mode_checkbox = gr.Checkbox(label="Update list", show_label = True, value=True, elem_classes=['small-small', 'icon-button'])    
                    
                    documents_dropdown = gr.Dropdown(choices = documents_list_string, multiselect=True, label="Select Documents")
                    Tools_mode_checkbox.change(fn=update_dropdown, inputs=[Tools_mode_checkbox, documents_dropdown], outputs=documents_dropdown)               
                    
                    Document_uploaded_file_pth = gr.UploadButton("UpLoad Text 🛠", file_types=[".txt, .epub, .pdf"], file_count="multiple") 
                    gr.Markdown("""Ingest a new document""", elem_classes=['left'])
                    state_msg = gr.Textbox(label="Status", visible=True)
                    Text_id = gr.Textbox(label="Text_id", visible=True)
                    Text_ids_list = gr.Textbox( label="Text_id list", visible=True)
                    documents_dropdown.select(fetch_titles_and_ids,[documents_dropdown],[Text_ids_list])
                    
                    
                    
                    csv_uploaded_file_pth = gr.UploadButton("UpLoad CSV 🛠", file_types=[".csv, .xls"], file_count="multiple")
                    gr.Markdown("""Ingest a new csv""", elem_classes=['left'])
                    csv_state_msg = gr.Textbox(label="Status", visible=True)
                    
                    csv_delimiter = gr.Textbox(',', placeholder="csv_delimiter", label = "CSV delimiter", info ="default=comma", elem_classes=['no-label', 'small-small'])
                    
                with gr.Tab("Sessions", elem_classes=['no-label', 'small-small']):    
                    session_id=gr.Textbox(label="SessionID: ",value=generate_new_session_id, elem_classes=['no-label', 'small-small'])
                    user_id = gr.Textbox(label="UID: ",value="Guest", elem_classes=['no-label', 'small-small'])    
                    limit_number = gr.Slider(minimum=0, maximum=1024, value=10, step=1, label="Limit reload", elem_classes=['no-label', 'small-small'])
                    
                    session_list_dropdown=gr.Dropdown(label="Session List", choices=Session_dropdown_list_string_init)
                    user_id.submit(fn=update_ddown_session,inputs=[user_id, session_list_dropdown], outputs=session_list_dropdown)   
                    

            with gr.Column(scale=10, elem_classes=['group-border']):
                chatbot_interface = gr.Chatbot(label="Chat output", elem_classes=['chuanhu_chatbot','no-label', 'small-big-textarea'], height=800, show_copy_button=True)
                
                with gr.Accordion("See picture", open=False):
                    Last_image_path_visibility = gr.Checkbox(value=False, visible=False)
                    with gr.Row():
                        Memorize_btn_2 = gr.Button()
                        Load_image_as_active_btn = gr.Button()
                    result_img = gr.Image("temp/1x1.png")
                with gr.Accordion("See table", open=False):
                    Last_table_path_visibility = gr.Checkbox(value=False, visible=False)
                    with gr.Row():
                        Memorize_btn_1 = gr.Button("Memorize")
                        Load_table_as_active_btn = gr.Button("Load as active")
                    result_table = gr.HTML("empty")
                chatbot_interface.change(check_output,None,[result_img, result_table]) 
                
                user_prompt = gr.Textbox(label="Input",scale=10)

    
    with gr.Tab("Sequence Design", elem_classes=['no-label', 'small-small']):
        
        with gr.Row():
           
            dataf = gr.Dataframe(load_df("promptchain/Default_Seq.csv"),wrap=True, col_count=[4,'fixed'])
           
        with gr.Row():
            move_up_btn = gr.Button("Row 🔼 Move Up", scale=1,elem_classes=['control-button'])
            move_down_btn = gr.Button("Row 🔽 Move Down", scale=1, elem_classes=['control-button'])
            del_btn = gr.Button("❌ Del Row", scale=1, elem_classes=['control-button'])
        with gr.Row():
            f_name = gr.Textbox(label="Filename",interactive=True)
            save_btn = gr.Button("💾 Save as .csv ", scale=1, elem_classes=['control-button'])
            save_status = gr.Textbox()
            statement = gr.Textbox(visible=False)
            
            dataf.select(on_select, None, statement)
            del_btn.click(delrow, inputs=[dataf, statement], outputs=[dataf])
            move_up_btn.click(move_up, inputs=[dataf, statement], outputs=[dataf])
            move_down_btn.click(move_down, inputs=[dataf, statement], outputs=[dataf])
            save_btn.click(save_to_csv, inputs=[dataf,f_name], outputs=[save_status])
    
    with gr.Tab("Converter", elem_classes=['no-label', 'small-small']):
        json_schema = gr.Textbox(label="Paste here a valid json schema", elem_classes=['no-label', 'small-small'])
        convert_json_btn = gr.Button("Convert in gbnf", scale=1,elem_classes=['control-button'])
        gbnf_schema = gr.Textbox(label="Here will appear the conversion in Gbnf", elem_classes=['no-label', 'small-small'])
        f_name_gbnf = gr.Textbox(label="Filename gbnf",interactive=True)
        save_gbnf_btn = gr.Button("💾 Save as .gbnf ", scale=1, elem_classes=['control-button'])
        save_gbnf_status = gr.Textbox()
        save_gbnf_btn.click(save_to_gbnf, inputs=[gbnf_schema,f_name_gbnf], outputs=[save_gbnf_status])
    
    with gr.Tab("Batch Execution", elem_classes=['no-label', 'small-small']):
        punchcard_file_pth = gr.UploadButton("Load batch punchcard sequence file", file_types=[".csv"], file_count="multiple")
        punchcard_filename = gr.Textbox(label="Active Prompt in chain Card")

        with gr.Row():
            punchcard_data = gr.Dataframe(wrap=True, col_count=[4,'fixed'])
        
        dataset_input_file_pth = gr.UploadButton("Load Dataset to handle", file_types=[".csv"], file_count="multiple")
        dataset_filename = gr.Textbox(label="Dataset Loaded")
        
        with gr.Row():
            dataset_input_data = gr.Dataframe(wrap=True, col_count=[1,'fixed'])

        run_punchcard_btn = gr.Button("Run PunchCard", elem_classes=['control-button'], scale=1)
        batch_output = gr.HTML(label="Batch output")
    
    with gr.Tab("Dataframe manipulation sequences", elem_classes=['no-label', 'small-small']):
        
        with gr.Row():
           
            csv_memory_df = gr.Dataframe(load_df("memory/memory.csv"),wrap=True, col_count=[4,'fixed'])
           
        with gr.Row():
            move_up_btn = gr.Button("Row 🔼 Move Up", scale=1,elem_classes=['control-button'])
            move_down_btn = gr.Button("Row 🔽 Move Down", scale=1, elem_classes=['control-button'])
            del_btn = gr.Button("❌ Del Row", scale=1, elem_classes=['control-button'])
            clear_btn=gr.Button("clear", scale=1, elem_classes=['control-button'])
        with gr.Row():
            f_name = gr.Textbox(label="Filename",interactive=True)
            save_btn = gr.Button("💾 Save as .csv ", scale=1, elem_classes=['control-button'])
            save_status = gr.Textbox()
            statement = gr.Textbox(visible=False)
            
            csv_memory_df.select(on_select, None, statement)
            del_btn.click(delrow, inputs=[csv_memory_df, statement], outputs=[csv_memory_df])
            move_up_btn.click(move_up, inputs=[csv_memory_df, statement], outputs=[csv_memory_df])
            move_down_btn.click(move_down, inputs=[csv_memory_df, statement], outputs=[csv_memory_df])
            save_btn.click(save_to_csv, inputs=[csv_memory_df,f_name], outputs=[save_status])
            clear_btn.click(clear_btn_csv,None,outputs=[csv_memory_df])
    with gr.Tab("Configuration", elem_classes=['no-label', 'small-small']):
        
        gpu_slice = gr.Slider(minimum=0, maximum=49, value=24, step=1, label="GPU slice", elem_classes=['no-label', 'small-small'])
        context_size = gr.Slider(minimum=1024, maximum=16384, value=4096, step=256, label="Context size", elem_classes=['no-label', 'small-small'])
        model_ddown_lst = gr.Dropdown(label="Available models in model dir ", choices=model_lst, max_choices=1)
        refresh_model_lst_btn = gr.Button("Refresh model list", scale=1, elem_classes=['control-button'])
        refresh_model_lst_btn.click(fn=update_ddown_model_lst,inputs=[model_ddown_lst], outputs=[model_ddown_lst])
        load_model_btn = gr.Button("Load model", scale=1, elem_classes=['control-button'])
                
        load_model_btn.click(load_model, inputs=[model_ddown_lst,gpu_slice,context_size], outputs=[title_md])
    
    with gr.Tab("Log", elem_classes=['no-label', 'small-small']):

        log_screen = gr.Textbox(label="Chat output", elem_classes=['chuanhu_chatbot','no-label', 'small-big-textarea'], lines=20, max_lines = 60)


    session_list_dropdown.select(retrieve_session,[user_id, session_list_dropdown, limit_number], [chatbot_interface])
    load_sequence_btn.click(load_sequence, inputs=[sequence_lst_ddown], outputs=[dataf,file_name, next_df_filename])
    
    Document_uploaded_file_pth.upload(Embed_txt_from_file, [Document_uploaded_file_pth], [state_msg, Text_id])
    user_prompt.submit(llm_answer_via_Tool,  inputs=[user_id, session_id, user_prompt, user_tag, assistant_tag, include_history, temperature, repeat_penalty, top_p, max_length, dataf, file_name, context_size], outputs=[chatbot_interface, log_screen, next_df_filename]).then(clear_user_prompt,[user_prompt],outputs=[user_prompt]).then(load_df_from_df,[next_df_filename],[dataf, file_name])
   
    csv_uploaded_file_pth.upload(upload_csv, [csv_uploaded_file_pth,csv_delimiter], [csv_state_msg, dataf, user_tag, assistant_tag, include_history, file_name]).then(llm_answer_via_Tool,  inputs=[user_id, session_id, user_prompt, user_tag, assistant_tag, include_history, temperature, repeat_penalty, top_p, max_length, dataf, file_name, context_size], outputs=[chatbot_interface, next_df_filename])
    Load_table_as_active_btn.click(SetNewUploadDf,None,[csv_state_msg])
    
    Chatbot_Mode_checkbox.change(fn=toggle_chat_mode, inputs=[Chatbot_Mode_checkbox, file_name, user_tag, assistant_tag], outputs=[user_tag, assistant_tag, dataf, include_history, file_name])
    reset_btn.click(reset_chat, [user_id, session_id],[chatbot_interface, session_id])
    back_btn.click(delete_last_interaction, [user_id, session_id],[chatbot_interface])
    convert_json_btn.click(generate_grammar_from_schema, json_schema,gbnf_schema)
    punchcard_file_pth.upload(load_df_from_file, [punchcard_file_pth], [punchcard_data,punchcard_filename])
    dataset_input_file_pth.upload(load_dataset_from_file, [dataset_input_file_pth], [dataset_input_data, dataset_filename])
    run_punchcard_btn.click(run_punchcard, inputs=[user_id, session_id,punchcard_data,user_tag, assistant_tag, include_history, temperature, repeat_penalty, top_p, max_length, dataset_input_data, context_size], outputs=[batch_output])
    btn_memorize.click(memorize, [user_id, session_id], [csv_memory_df], preprocess=False)

demo.queue()
#demo.launch()
demo.launch(share=True)

