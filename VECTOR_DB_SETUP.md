# Vector Database Setup Guide

This guide explains how to set up the vector database connection for different environments.

## Environment Configuration

The application uses environment variables to determine how to connect to the vector database. The key variable is `VECTOR_DB_HOST`, which should be set differently depending on your environment:

### Running Inside Docker

When running inside Docker containers (using docker-compose):

```
VECTOR_DB_HOST=vector-db
```

This works because Docker's internal DNS resolves the service name `vector-db` to the appropriate container IP address.

### Running Locally (Outside Docker)

When running directly on your host machine (not in Docker):

```
VECTOR_DB_HOST=localhost
```

This connects to the vector database through the port mapping defined in your docker-compose.yml file.

## Setting Up Different Machines

1. **Create a `.env` file** on each machine where you deploy the application
2. **Set the appropriate `VECTOR_DB_HOST` value** based on how you'll run the application:
   - For Docker deployment: `VECTOR_DB_HOST=vector-db`
   - For local development: `VECTOR_DB_HOST=localhost`

## Troubleshooting Connection Issues

If you encounter connection issues, use the included diagnostic script:

```bash
python check_vector_db.py
```

This script will:
- Check if the hostname can be resolved
- Verify if the port is accessible
- Test the connection to the vector database
- Provide suggestions if any issues are found

### Common Issues and Solutions

1. **"Temporary failure in name resolution" error**:
   - If running outside Docker, ensure `VECTOR_DB_HOST=localhost` in your `.env` file
   - If running inside Docker, ensure `VECTOR_DB_HOST=vector-db` in your `.env` file

2. **Connection timeout**:
   - Ensure the vector database container is running: `docker-compose ps`
   - Check if the port is correctly mapped in docker-compose.yml
   - Try restarting the containers: `docker-compose down && docker-compose up -d`

3. **Port not accessible**:
   - Check if another service is using the same port
   - Verify the port mapping in docker-compose.yml

## Verifying the Setup

After configuring your environment, run the pipeline to verify everything works:

```bash
python -m app.pipeline --pdf-dir ./data/documents --rebuild
```

If successful, you should see a message indicating that vectors were uploaded to the database. 