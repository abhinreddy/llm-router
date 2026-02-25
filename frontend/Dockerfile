FROM node:20-alpine

WORKDIR /app

# Install dependencies (cached layer)
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

EXPOSE 5173

# --host binds Vite to 0.0.0.0 so the container port is reachable
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
