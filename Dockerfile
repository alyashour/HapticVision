# Use an official Python runtime as a base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /Users/alya/PycharmProjects/HandCV

# Copy the requirements file
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Command to run your app (adjust as necessary)
CMD ["python", "GUI/main.py"]
