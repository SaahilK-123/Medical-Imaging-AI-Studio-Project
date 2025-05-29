## U-Net Implementation with Keras 3.0 and TensorFlow in Python

This project implements the U-Net Architecture using the Keras 3.0 Deep Machine Learning Library with TensorFlow as its backend.

## Full Stack Application

The FastAPI backend is deployed on [Render](https://render.com) after being complied into a Docker image. The Frontend application is deployed on [Streamlit Community Cloud](https://etl-tutorial.streamlit.app) (Python version) and [Vercel](https://etl-tutorial.vercel.app) (HTML version). The purpose of this application is to show different data privileges for different users: Data Manager and Employee where the manager has more data access while the employee can only access personal information.

- The username and password for manager are: manager1 - managerpass
- There are 3 employees base on the SQL data: ann, bob, and john with the same password 1234

Please follow the [Tutorial Videos](https://www.youtube.com/watch?v=yIflPdERBvY&list=PL5WVbgdMDtS0rBQzayZnYlnIDHHyEAYxy) for project set up and implementation

To run the application locally, follow these steps:
- Create .env file in the project directory and add the required environment variables
- Create a virtual environment and install the required packages: pip install -r requirements.txt
- Run the python backend: **uvicorn utils.api:app**
- Change the const api_url variable in webapp/main.js to http://127.0.0.1:8000
- Open the **index.html** file in your browser or run command **streamlit run webapp/app.py** and use the app 

## Required Environment varables:
- ACCOUNT_STORAGE="YOUR STORAGE ACCOUNT"
- USERNAME_AZURE="YOUR SQL USERNAME"
- PASSWORD="YOUR SQL PASSWORD"
- SERVER="YOUR AZURE SQL SERVER" * MAKE SURE TO HAVE database.windows.net
- DATABASE="YOUR AZURE SQL DATABASE"
- JWT_SECRET_KEY="ANY SECRET KEY FOR THE APP"

### Official Azure Documentations:

[Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?tabs=managed-identity%2Croles-azure-portal%2Csign-in-visual-studio-code&pivots=blob-storage-quickstart-scratch&fbclid=IwAR0_SXxKXmnzjU8YgZ7xHys0-F2yG-V4pXQk8us7wv1Z-gEys62RS6ODBRg#prerequisites)

[Azure SQL database](https://learn.microsoft.com/en-us/azure/azure-sql/database/azure-sql-python-quickstart?view=azuresql&tabs=windows%2Csql-inter)

### Core Prerequisites:

Azure account with an active subscription - [create an account for free](https://azure.microsoft.com/en-us/free/?ref=microsoft.com&utm_source=microsoft.com&utm_medium=docs&utm_campaign=visualstudio)

Azure Storage account - [create a storage account](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)

An Azure SQL database configured with Microsoft Entra authentication. You can create one using the [Create database quickstart](https://learn.microsoft.com/en-us/azure/azure-sql/database/single-database-create-quickstart?view=azuresql&tabs=azure-portal).

The latest version of the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/get-started-with-azure-cli).

Visual Studio Code with the Python extension.

Python 3.8 or later. If you're using a Linux client machine, see [Install the ODBC driver](https://learn.microsoft.com/en-us/sql/connect/python/pyodbc/step-1-configure-development-environment-for-pyodbc-python-development?view=sql-server-ver16&tabs=linux#install-the-odbc-driver).
