FROM python
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
ENV STREAMLIT_APP=streamlit_app.py
CMD ["streamlit", "run", "--client.showErrorDetails", "False", "streamlit_app.py"]