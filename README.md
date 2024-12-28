# FrameWise: AI Framework Selector

FrameWise is a tool designed to help you select the most suitable AI framework for your specific use case. It evaluates frameworks based on criteria such as throughput, latency, scalability, security, ease of use, model support, and cost efficiency.

## Objective

The objective of FrameWise is to provide a structured and data-driven approach to selecting the best AI framework for your project. By analyzing your use case and evaluating frameworks against predefined criteria, FrameWise ensures that you make an informed decision.

## Features

- **Use Case Analysis**: Evaluate your use case against predefined criteria.
- **Framework Comparison**: Compare popular AI frameworks like SGLang, NVIDIA NIM, vLLM, Mistral.rs, and FastChat.
- **Criteria-Based Selection**: Select frameworks based on throughput, latency, scalability, security, ease of use, model support, and cost efficiency.
- **Customizable Input**: Add your own use case description if it’s not in the pre-populated list.

## Installation

Follow these steps to set up FrameWise on your local machine.

### Prerequisites

- Python 3.11 or higher
- Git (optional, for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/KingLeoJr/FrameWise.git
cd FrameWise
Step 2: Create a Virtual Environment
Create a virtual environment to isolate dependencies:

bash
Copy
python -m venv venv
Activate the virtual environment:

On Windows:

bash
Copy
venv\Scripts\activate
On macOS/Linux:

bash
Copy
source venv/bin/activate
Step 3: Install Dependencies
Install the required Python packages:

bash
Copy
pip install -r requirements.txt
Step 4: Set Up the .env File
Create a .env file in the root directory of the project and add your environment variables. For example:

plaintext
Copy
API_KEY=your_api_key_here
Replace your_api_key_here with your actual API key.

Step 5: Run the Application
Start the Streamlit application:

bash
Copy
streamlit run app.py
Open your browser and navigate to the provided URL (usually http://localhost:8501).

Usage
Select a Use Case: Choose a use case from the dropdown or enter your own.

Submit: Click the "Submit" button to evaluate the most suitable AI framework.

View Results: The recommended framework and evaluation criteria will be displayed in a table.

Contributing
Contributions are welcome! If you’d like to contribute, please follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeatureName).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeatureName).

Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Thanks to the Streamlit team for the amazing framework.

Special thanks to the open-source community for their contributions.

For any questions or issues, please open an issue on the GitHub repository.

Copy

---

### **How to Use**

1. **Copy the Entire Block**:
   - Select the entire content above and copy it.

2. **Paste into `README.md`**:
   - Open a text editor (e.g., Notepad, VS Code).
   - Create a new file and paste the copied content.
   - Save the file as `README.md` in the root directory of your project.

3. **Add and Commit the `README.md` File**:
   ```bash
   git add README.md
   git commit -m "Add README.md file"
   git push origin main