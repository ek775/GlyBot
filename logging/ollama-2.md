/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/langchain/_api/module_import.py:92: LangChainDeprecationWarning: Importing ChatMessageHistory from langchain.memory is deprecated. Please replace deprecated imports:

>> from langchain.memory import ChatMessageHistory

with new imports of:

>> from langchain_community.chat_message_histories import ChatMessageHistory
You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here https://python.langchain.com/v0.2/docs/versions/v0_2/ 
  warn_deprecated(
Setting up new vector DB...









































































Traceback (most recent call last):
  File "/home/ek224/Desktop/GlyBot/RAG_core.py", line 63, in <module>
    db = QdrantSetup(data_dir='./textbook_text_data/', cache=cache, name=name)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/Desktop/GlyBot/pipelines/vector_store.py", line 43, in __init__
    self._initialize_vector_db(
  File "/home/ek224/Desktop/GlyBot/pipelines/vector_store.py", line 127, in _initialize_vector_db
    pipeline.run(documents=documents) 
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/llama_index/core/ingestion/pipeline.py", line 734, in run
    nodes = run_transformations(
            ^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/llama_index/core/ingestion/pipeline.py", line 124, in run_transformations
    nodes = transform(nodes, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/llama_index/core/base/embeddings/base.py", line 439, in __call__
    embeddings = self.get_text_embedding_batch(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/llama_index/core/instrumentation/dispatcher.py", line 223, in wrapper
    result = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/llama_index/core/base/embeddings/base.py", line 331, in get_text_embedding_batch
    embeddings = self._get_text_embeddings(cur_batch)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/llama_index/embeddings/ollama/base.py", line 64, in _get_text_embeddings
    embeddings = self.get_general_text_embedding(text)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/llama_index/embeddings/ollama/base.py", line 89, in get_general_text_embedding
    response = requests.post(
               ^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/requests/api.py", line 115, in post
    return request("post", url, data=data, json=json, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/requests/api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/requests/adapters.py", line 667, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/urllib3/connectionpool.py", line 793, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/urllib3/connectionpool.py", line 537, in _make_request
    response = conn.getresponse()
               ^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/site-packages/urllib3/connection.py", line 466, in getresponse
    httplib_response = super().getresponse()
                       ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/http/client.py", line 1428, in getresponse
    response.begin()
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/http/client.py", line 331, in begin
    version, status, reason = self._read_status()
                              ^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/http/client.py", line 292, in _read_status
    line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ek224/anaconda3/envs/glybot/lib/python3.12/socket.py", line 707, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt