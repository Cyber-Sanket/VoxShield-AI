# VoxShield AI – Static Landing Page

A modern, responsive, cyber-themed static landing page for **VoxShield AI – AI-Powered Audio Deepfake Detection System** by team **Ghost Protocol**.

## Features

- **Cybersecurity Dark Aesthetic:** Deep navy canvas (`#050811`), neon green (`#00FFA3`), electric cyan (`#00E5FF`), and blue accents.
- **Interactive Audio Spectrum Simulator:** Real-time HTML5 Canvas animation comparing genuine human voice formants against synthetic vocoder frequency cutoffs and buzz artifacts.
- **Configurable Application Endpoint:** Easily link the landing page to your local Streamlit instance (`http://localhost:8501`) or cloud deployment (`https://voxshield-ai.streamlit.app`) through the built-in modal or `script.js`.
- **Pure Vanilla Stack:** Built strictly with HTML5, CSS3, and modern JavaScript. Zero build steps, zero node dependencies.
- **Mobile Responsive:** Seamless experience across desktop, tablet, and mobile devices.

---

## Deployment to Netlify

### Option 1: Drag-and-Drop (Zero Config)
1. Log into your [Netlify](https://app.netlify.com/) account.
2. Drag and drop the `netlify-site` folder directly into the Netlify "Sites" dashboard.
3. Your site will be deployed instantly with an SSL-enabled `.netlify.app` domain.

### Option 2: Netlify CLI
```bash
# Navigate to the project folder
cd netlify-site

# Deploy via Netlify CLI
npx netlify-cli deploy --prod
```

### Option 3: GitHub Continuous Deployment
1. Push your repository to GitHub (`https://github.com/Cyber-Sanket/VoxShield-AI`).
2. In Netlify, click **"Add new site" -> "Import an existing project"**.
3. Set the **Base directory** to `netlify-site`.
4. Leave build command empty and publish directory as `.`.
5. Click **"Deploy VoxShield AI"**.

---

## Configuring the Streamlit Application URL

By default, all "Launch Application" buttons target `http://localhost:8501`.

To point the landing page to your hosted Streamlit app:
1. Open `netlify-site/script.js`.
2. Update the `CONFIG` object at line 14:
   ```javascript
   const CONFIG = {
       DEFAULT_STREAMLIT_URL: "https://your-deployed-app.streamlit.app",
       GITHUB_REPO_URL: "https://github.com/Cyber-Sanket/VoxShield-AI"
   };
   ```
3. Alternatively, any visitor or evaluator can click the **"⚙️ Configure"** button right on the live page to change the endpoint dynamically for their session!
