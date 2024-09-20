

1. Building containers:
   To build your container for the first time:
   ```
   export $(cat ./server/.env | xargs) && docker-compose up --build
   ```
   To build your container with `langchain_service` for the first time:
   ```
   export $(cat ./server/.env | xargs) && export $(cat ./langchain_service/.env | xargs) && docker-compose up --build
   ```
   To start your containers without rebuilding:
   ```
   docker-compose up
   ```

   To run the containers in detached mode (in the background):
   ```
   docker-compose up -d
   ```

   To stop the containers:
   ```
   docker-compose down
   ```

2. Exporting environment variables:
   Once you've built your containers with the environment variables, those variables are stored within the container's environment. You don't need to export them again each time you start the containers.

   However, if you make changes to your .env file and want those changes to take effect, you would need to rebuild the containers:
   ```
   export $(cat ./server/.env | xargs) && docker-compose up --build
   ```

   If you don't want to rebuild but want to update environment variables, you can use:
   ```
   docker-compose up --env-file ./server/.env
   ```

Some additional tips:

1. Viewing logs:
   If your containers are running in detached mode and you want to see the logs:
   ```
   docker-compose logs
   ```
   Or for a specific service:
   ```
   docker-compose logs server
   ```

2. Entering a running container:
   If you need to run commands inside a running container:
   ```
   docker-compose exec server sh
   ```
   This opens a shell in the server container.

3. Restarting a single service:
   If you've made changes to just one part of your application:
   ```
   docker-compose restart server
   ```

4. Checking the status of your containers:
   ```
   docker-compose ps
   ```