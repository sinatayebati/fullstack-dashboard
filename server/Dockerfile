# # DEV CONTAINER CONFIG
# Select a base image
FROM node:20.11.1-alpine3.19

# Create a directory and go to directory 
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Install dependencies including development dependencies
RUN npm install

# Install nodemon globally for development
RUN npm install -g nodemon

# Open the port for express server
EXPOSE 5001

# Add a healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:5001/health', (res) => res.statusCode === 200 ? process.exit(0) : process.exit(1))"

# Use nodemon to run the server and watch for changes
CMD ["nodemon", "index.js"]




# # PROD CONTAINER CONFIG
# # Select a base image
# FROM node:20.11.1-alpine3.19

# # Create a directory and go to directory 
# WORKDIR /app

# # Copy the package.json file to my current directory for install necessary dependence  
# COPY package.json .

# # Install the dependence
# RUN npm install

# # Copy other files to my current directory
# COPY . .

# # Open the port for express server
# EXPOSE 5001

# # Add a healthcheck
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#   CMD node -e "require('http').get('http://localhost:5001/health', (res) => res.statusCode === 200 ? process.exit(0) : process.exit(1))"

# # Run express server in foreground
# CMD ["npm", "start"]