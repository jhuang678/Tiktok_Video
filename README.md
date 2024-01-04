# Data Mining on TikTok Video and User Engagement

*In a digital era where video content dominates, this report investigates methods to verify TikTok video content and enhance user engagement. By analyzing over 19,000 videos, we have identified methods to detect potentially misleading content and understand what drives user interaction. Our work is pivotal for digital advertising, platform governance, and the creation of a trustworthy social media environment.*

## Table of Contents
- [Introduction](#introduction)
- [Data Sources](#data-sources)
- [Proposed Methodology](#proposed-methodology)
- [Analysis and Results](#analysis-and-results)
- [Conclusions](#conclusions)

## Introduction
This report explores TikTok's content authenticity and user interaction patterns using data-driven techniques, highlighting its impact on digital communication and advertising.

### Motivation
The motivations driving this research are manifold, focusing on enhancing user experience and trust within TikTok's platform:
- Ensuring **content integrity** to establish TikTok as a source of reliable information.
- Improving **advertising effectiveness** through detailed analysis of user engagement metrics.
- Advancing **content recommendation algorithms** for personalized user experiences.

### Problem Statement and Solving Strategies
We address the following key questions to navigate the challenges of data mining in the context of TikTok:
- **Classification of Content:** Development and implementation of algorithms to discern and categorize video content based on authenticity.
- **Engagement Analysis:** Utilization of clustering techniques to segment videos by user interaction levels, revealing patterns that drive engagement.
- **Share-ability Assessment:** Application of NLP to determine the share-ability of content, identifying what makes certain videos more viral than others.

### Data Mining Challenges
In exploring TikTok's user engagement and content reliability, we faced multiple challenges:
- **Data Complexity:** Managing the volume, diversity, and quality of data.
- **Model Precision and Bias:** Developing accurate models while avoiding bias and over-fitting.
- **Broad Applicability:** Ensuring findings and methodologies are adaptable across various social media platforms.

## Data Sources
### Overview of the Data Set
The dataset, sourced from Kaggle, includes 19,382 TikTok videos, categorized as 'claim' or 'opinion', with a range of engagement metrics. Details: [Kaggle Dataset](https://www.kaggle.com/datasets/yakhyojon/tiktok/data).

### Composition of the Dataset
Our analysis begins with an expansive dataset comprising 19,382 TikTok videos, each encapsulating various facets of user engagement. Each video in the dataset is described by 12 distinct attributes, which include both quantitative engagement metrics and categorical descriptors for content type.

### Attributes of Interest
For the full list of attributes, please refer to the original report.

### Initial Data Exploration
An initial glimpse into the dataset showcases the diversity of content and engagement. (Here, you may want to describe the key points or upload the tables as images and link them in the README.)

#### Table: Introduction of the Data Columns
Column Name | Type | Description
------------|------|------------
\# | int | TikTok assigned number for video with claim/opinion.
claim_status | obj | Whether the published video has been identified as an “opinion” or a “claim.”
video_id | int | Random identifying number assigned to video.
video_duration_sec | int | Video duration in seconds.
video_transcription_text | obj | Transcribed text of the words spoken in the published video.
verified_status | obj | The status of the user who published the video in terms of their verification.
author_ban_status | obj | The status of the user who published the video in terms of their permissions.
video_view_count | float | Total number of times the published video has been viewed.
video_like_count | float | Total number of times the published video has been liked by other users.
video_share_count | float | Total number of times the published video has been shared by other users.
video_download_count | float | Total number of times the published video has been downloaded by other users.
video_comment_count | float | Total number of comments on the published video.

#### Table: Summary of Video Data
\# | claim_status | video_id | video_duration_sec | video_transcription | verified_status | author_ban_status | video_view_count | video_like_count | video_share_count | video_download_count | video_comment_count
---|--------------|----------|--------------------|---------------------|-----------------|-------------------|------------------|------------------|------------------|---------------------|--------------------
0 | claim | 7017666017 | 59 | ... | not verified | under review | 343296.0 | 19425.0 | 241.0 | 1.0 | 0.0
1 | claim | 4014381136 | 32 | ... | not verified | active | 140877.0 | 77355.0 | 19034.0 | 1161.0 | 684.0
2 | claim | 9859838091 | 31 | ... | not verified | active | 902185.0 | 97690.0 | 2858.0 | 833.0 | 329.0
3 | claim | 1866847991 | 25 | ... | not verified | active | 437506.0 | 239954.0 | 34812.0 | 1234.0 | 584.0
4 | claim | 7105231098 | 19 | ... | not verified | active | 56167.0 | 34987.0 | 4110.0 | 547.0 | 152.0
![image](https://github.com/jhuang678/Tiktok_Video/assets/100253011/ed9da2f7-f052-4deb-83e9-58bde0414a26)



### Missing Values and Duplicates
To ensure the reliability of our findings, we meticulously addressed issues of missing information and redundancy:
- **Handling Missing Values:** Identified and removed missing values from the dataset.
- **Addressing Duplicate Records:** Confirmed no duplicate values, indicating a unique dataset.

### General Exploratory Data Analysis (EDA)
#### Categorical Data
Categorical data variables play a crucial role in understanding the characteristics of the TikTok data set. Our exploratory data analysis revealed the following distributions. (You might consider summarizing the key findings or including images of the charts.)

#### Numerical Data
The numerical data in our dataset primarily comprises video duration and various engagement metrics. We generated histograms and box plots for each attribute to observe their distributions. (Summarize or link to visual representations.)

## Proposed Methodology
Our methodology includes data collection and cleaning, exploratory data analysis, data processing, standardization, and applications of classification, clustering, and topic modeling techniques to derive insights.

### Data Mining Process
The data analysis for this study is partitioned into several sequential stages, each building on the findings of the previous one. (Elaborate on the stages or use a diagram to illustrate.)

### Research Questions
We explore three main research questions regarding classifying claims and opinions, clustering user engagement metrics, and analyzing topic clustering and share-ability. (Detail each question and the approach taken.)

## Analysis and Results
### Research Question 
