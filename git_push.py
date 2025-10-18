# Tell git who you are (do this once)
git config --global user.name "decoder-architect"
git config --global user.email "decoder_architect@proton.me"

# Initialize your local repository
git init

# Link it to GitHub (replace URL with your repo link)
git remote add origin https://github.com/decoder-architect/omega-aurora-codex.git

# Add all current files
git add .

# Make your first commit
git commit -m "Initial Aurora Codex upload"

# Push everything to GitHub
git push -u origin main
