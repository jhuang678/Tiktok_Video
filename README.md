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

Note: regarding claim status, an “opinion” refers to an individual’s or group’s personal belief or thought.
A “claim” refers to information that is either unsourced or from an unverified source.

### Initial Data Exploration
An initial glimpse into the dataset showcases the diversity of content and engagement.

#### Table: Summary of Video Data
\# | claim_status | video_id | video_duration_sec | video_transcription | verified_status | author_ban_status | video_view_count | video_like_count | video_share_count | video_download_count | video_comment_count
---|--------------|----------|--------------------|---------------------|-----------------|-------------------|------------------|------------------|------------------|---------------------|--------------------
0 | claim | 7017666017 | 59 | ... | not verified | under review | 343296.0 | 19425.0 | 241.0 | 1.0 | 0.0
1 | claim | 4014381136 | 32 | ... | not verified | active | 140877.0 | 77355.0 | 19034.0 | 1161.0 | 684.0
2 | claim | 9859838091 | 31 | ... | not verified | active | 902185.0 | 97690.0 | 2858.0 | 833.0 | 329.0
3 | claim | 1866847991 | 25 | ... | not verified | active | 437506.0 | 239954.0 | 34812.0 | 1234.0 | 584.0
4 | claim | 7105231098 | 19 | ... | not verified | active | 56167.0 | 34987.0 | 4110.0 | 547.0 | 152.0

Note: Regarding video transcription text, as this assignment does not incorporate text analysis, the
value presented in the preceding table has been streamlined.

### Missing Values and Duplicates
To ensure the reliability of our findings, we meticulously addressed issues of missing information and redundancy:
- **Handling Missing Values:** Identified and removed missing values from the dataset.
- **Addressing Duplicate Records:** Confirmed no duplicate values, indicating a unique dataset.

After removing 298 entries with missing data and confirming no duplicates, the dataset comprises
19,084 unique videos for analysis.

### General Exploratory Data Analysis (EDA)
#### Categorical Data
Categorical data variables play a crucial role in understanding the characteristics of the TikTok data set. Our exploratory data analysis revealed the following distributions. (You might consider summarizing the key findings or including images of the charts.)

![Pie Chart](img/pie_chart.png)

- **Claim Status:** The data set is almost evenly split between videos labeled as 'opinion' and 'claim', with 49.65% and 50.35% respectively, signifying a balanced representation of content types.
- **Verified Status:** A large majority of video authors, 93.71%, are not verified, while only a small fraction, 6.29%, have verified status. This suggests that the platform is predominantly used by non-verified users.
- **Author Ban Status:** Most authors are active, constituting 80.61% of the data set. A smaller percentage of authors are 'banned' (8.57%) or 'under review' (10.83%), indicating moderate levels of content moderation.

#### Numerical Data
The numerical data in our dataset primarily comprises video duration and various engagement metrics.

Histograms were generated for each attribute in the data set to observe their distributions:
![Histogram of Feature](img/histogram.png)

Box plots were generated for each attribute in the data set:
![Box Plot of Features](img/boxplot.png)

Our exploratory data analysis (EDA) of these quantities has yielded key insights:
- **Video Duration:** The duration of the videos shows a uniform distribution, indicating a wide variety of content lengths with no apparent extremes or biases towards shorter or longer videos.
- **Engagement Metrics:** Metrics such as views, likes, shares, downloads, and comments are heavily right-skewed. This suggests that while most videos receive a low level of engagement, there are a few videos that achieve exceptionally high engagement, standing out as outliers.

These findings are crucial as they suggest that most content struggles to achieve virality, with only a few videos breaking through to significant popularity. 

## Proposed Methodology
Our methodology includes data collection and cleaning, exploratory data analysis, data processing, standardization, and applications of classification, clustering, and topic modeling techniques to derive insights.

### Data Mining Process
The data analysis for this study is partitioned into several sequential stages, each building on the findings of the previous one. (Elaborate on the stages or use a diagram to illustrate.)
![Data Mining Process](img/dm_process.png)

Note:  The top flow relates to classifying claim or opinion shorts, the middle flow relates to grouping videos by user engagement, and the bottom flow focuses on identifying topics that enhance share-ability.

The detail explanation of each stage is as follows:
- **Data Collection:** Sourced over 19,000 TikTok video records from Kaggle.
- **Data Cleaning:** Removed missing values and duplicates for a clean dataset.
- **General EDA:** Broad analysis of categorical and numerical variables.
- **Data Pre-processing:** Prepared data for analysis with normalization and transformation.
- **EDA:** In-depth analysis using visualizations to identify patterns.
- **Standardization:** Normalized data to a common scale for machine learning.
- **Classification Modeling:** Created models to classify video content authenticity.
- **Cluster Modeling:** Used clustering to segment videos by user interactions.
- **Topic Modeling:** Employed NLP to uncover themes in video transcriptions.

These stages are designed to incrementally advance our understanding, culminating in actionable insights into the drivers of user engagement on TikTok.

### Research Question 1: Classifying Claims and Opinions

Our method for classifying TikTok videos into claims or opinions includes:
- **Feature Selection:** Focusing on claim status, video duration, user verification, author status, and engagement metrics.
- **Visualization Techniques:** Using pie charts, pair plots, KDE plots, heat maps, and histograms.
- **Models for Classification:** Implementing Random Forest, Gradient Boosting, KNN, LDA, Logistic Regression, and Decision Tree.
- **Challenges:** Addressing multicollinearity, hyper-parameter tuning, and over-fitting.
- **Process Stages:** Covering data pre-processing, EDA, standardization, classification modeling, and result interpretation.

This streamlined process aims to accurately categorize video content on TikTok.

### Research Question 2: Clustering User Engagement Metrics

Our approach to understanding user engagement patterns on TikTok involves:
- **Feature Selection:** Analyzing metrics like video views, likes, shares, downloads, and comments.
- **Visualization Techniques:** Utilizing the Elbow diagram for optimal cluster identification in K-means.
- **Models for Clustering:** Applying K-means for unsupervised clustering.
- **Challenges:** Determining the right number of clusters and ensuring adequate iteration for optimal clustering.
- **Process Stages:** Involving data preprocessing, EDA, standardization, cluster modeling, and result interpretation.

This method aims to segment TikTok videos into distinct engagement groups for deeper insights.

### Research Question 3: Topic Clustering and Share-ability

Our approach for analyzing the share-ability of TikTok videos involved:
- **Feature Selection and Preprocessing:** Focusing on video transcriptions and related metrics.
- **Visualization and Modeling:** Employing word clouds, sentiment analysis, and Latent Dirichlet Allocation (LDA) for topic modeling.
- **Challenges and Goals:** Addressing text analysis complexity and identifying key topics driving share-ability.

This approach is designed to identify the thematic elements that contribute to the viral nature of TikTok videos, contributing to a deeper understanding of content share-ability on the platform.

### Conclusion of Methodology

Our methodical approach ensures a systematic exploration of the TikTok data set, with the ultimate goal of unveiling the intricacies of content engagement. By the end of this process, we aim to offer a clear narrative of what drives user interactions and share-ability on TikTok, providing valuable insights for content creators and platform moderators alike.

