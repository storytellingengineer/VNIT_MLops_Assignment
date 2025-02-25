# Use the official Python image as a base image
FROM python:3.11-slim
# Set the working directory in the container
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt .
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application files
COPY . .
# Expose the port Fast runs on
EXPOSE 5003
# Set environment variables
ENV FAST_API_APP=app.py
ENV FAST_RUN_HOST=0.0.0.0
ENV FAST_APP_PORT=5003
# Command to run the application
CMD ["uvicorn", "app:app", "--host ${FAST_RUN_HOST}", "--port ${FAST_APP_PORT}", "--reload"]