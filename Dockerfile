FROM python
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
#RUN docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
COPY . /app
ENV STREAMLIT_APP=streamlit_app.py
CMD ["streamlit", "run", "--client.showErrorDetails", "False", "streamlit_app.py"]