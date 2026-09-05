FROM condaforge/mambaforge:latest

WORKDIR /app

# Install system tools for port management
RUN apt-get update && apt-get install -y lsof psmisc procps vim nano && rm -rf /var/lib/apt/lists/*

# Copy and install from your exact working environment
COPY environment.yml .
RUN mamba env update -n base -f environment.yml

# Install PROJ data and set environment variable
RUN mamba install -y proj-data -c conda-forge
ENV PROJ_LIB=/opt/conda/share/proj

COPY . /app

EXPOSE 6060
EXPOSE 8080

CMD ["python", "main.py"]
