# # DEV CONTAINER CONFIG
# Use a Node.js base image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of your app's source code
COPY . .

# Expose port 3000 for the development server
EXPOSE 3000

# Start the app in development mode
CMD ["npm", "start"]


# # PROD CONTAINER CONFIG
# # Stage-1 & spacify a name 'builder'
# FROM  node:latest AS builder

# # Create a directory  and go to directory 
# WORKDIR /app

# # Copy the package.json file to my current directory for install necessary dependence  
# COPY package.json .

# # Install the dependence
# RUN npm install

# # Copy other file to my current directory
# COPY . .

# # Build optimise static file
# RUN npm run build

# # Stage-2
# FROM nginx:1.25.2-alpine-slim

# # Copy static file to my nignx folder for serve static contain
# COPY --from=builder /app/build /usr/share/nginx/html

# # Open the port for react
# EXPOSE 80

# # Run nginx in foreground
# CMD ["nginx", "-g", "daemon off;"]