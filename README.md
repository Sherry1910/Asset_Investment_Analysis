 📊 Advanced Qualitative Analysis on TAMIS for Asset Management

Unlock insights from Tanzania's TAMIS platform by turning complex fund performance data into structured, actionable intelligence.

🔍 Overview

This project automates the extraction and analysis of investment fund data from the [TAMIS website](https://uttamis.co.tz/fund-performance). By leveraging Python, Selenium, Parsel, and BeautifulSoup, it captures paginated and dynamic tables, cleans and structures the data, and prepares it for financial analysis and strategic asset management.

 💡 Why This Matters

Fund managers and analysts often face challenges accessing real-time, structured data for informed decision-making. This tool bridges that gap by:

- Analyze qualitative data such as policy documents, performance reports, and strategy briefs from TAMIS.
- Identify investment trends, thematic priorities, and asset allocation narratives.
- Provide actionable insights for optimizing fund management decisions.
- Visualize thematic patterns and strategic insights using data storytelling techniques.

 ⚙️ Features

- 🔄 Automated Web Scraping of paginated fund performance tables  
- 🧼 Data Cleaning & Structuring into CSV formats  
- 📁 Ready-to-Use Output for integration into dashboards or financial models  
- 🛡️ Error Handling for smooth navigation through dynamic web content

 🛠 Tech Stack

- Python 3.x  
- Selenium WebDriver  
- BeautifulSoup  
- Pandas  
- ChromeDriver
- Parsel
- Jupyter Notebook IDE
- Power BI for interactive dashboards


 🗂️ Project Structure
📁 uttamis_analysis/

├── RawData.py # Main script to scrape and process data
├── Desiredata 19072024.csv # Output: cleaned dataset in tabular format
├── Prediction.py # Python script to forecast the stock prices
├── ML_UTT.py # Script to apply machine learning models for forecast and handling stationarity
├── Designed a dashboard to display the trend and return on investment for the investors
├── Report of the project
├── README.md # Project overview and setup instructions

 🏁 Conclusion

The project empowers asset managers and stakeholders to interpret qualitative signals from the TAMIS platform more effectively, contributing to better-informed decisions in fund allocation, performance monitoring, and long-term investment strategy.

 📂 Output Files

- Key themes and focus areas in asset management policies
- Insightful visualizations summarizing qualitative findings
- Investment recommendations based on policy alignment and trends
- Report summarizing the implications for strategic planning

 🚀 How to Use

1. Install dependencies
   ```bash
   pip install -r requirements.txt
2. Run the script:
   python RawData.py, Prediction.py, ML_UTT.py
   

📜 License

This project is open for educational and research purposes.
