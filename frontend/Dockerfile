# ============================================================
# Auslan Sign Retrieval System — Frontend Dockerfile (Next.js)
# ============================================================
# Multi-stage build:
#   Stage 1 (deps)     — install node_modules
#   Stage 2 (builder)  — next build (standalone output)
#   Stage 3 (runner)   — minimal Node.js runtime
#
# Build:
#   docker build -t auslan-frontend ./frontend
# Run (standalone):
#   docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://localhost:8000 auslan-frontend
# ============================================================

# ----------------------------------------------------------------
# Stage 1: Install dependencies
# ----------------------------------------------------------------
FROM node:20-alpine AS deps

RUN apk add --no-cache libc6-compat

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --frozen-lockfile

# ----------------------------------------------------------------
# Stage 2: Build
# ----------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build-time env vars (baked into the bundle)
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Enable standalone output for minimal image size
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ----------------------------------------------------------------
# Stage 3: Minimal runtime
# ----------------------------------------------------------------
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs && \
    adduser  --system --uid 1001 nextjs

# Copy standalone build artefacts
COPY --from=builder /app/public        ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static     ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -qO- http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
