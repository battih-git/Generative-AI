from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash",)

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task = 'text-generation')
model2 = ChatHuggingFace(llm=llm)

prompt_1 = PromptTemplate(
    template='Generate short and simple notes from the text \n {text}',
    input_variables=['text']
)

prompt_2 = PromptTemplate(
    template='Generate 5 shorter questions answers from the following text \n {text}',
    input_variables=['text']
)

prompt_3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n {notes} \n {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt_1 | model1 | parser,
    'quiz' : prompt_2 | model2 | parser,
}
)


merge_chain = prompt_3 | model1 | parser

chain = parallel_chain | merge_chain

text = '''
Logistic Regression is a supervised machine learning algorithm used for classification problems. Unlike linear regression which predicts continuous values it predicts the probability that an input belongs to a specific class.

    It is used for binary classification where the output can be one of two possible categories such as Yes/No, True/False or 0/1.
    It uses sigmoid function to convert inputs into a probability value between 0 and 1.

    Types of Logistic Regression

Logistic regression can be classified into three main types based on the nature of the dependent variable:

    Binomial Logistic Regression: This type is used when the dependent variable has only two possible categories. Examples include Yes/No, Pass/Fail or 0/1. It is the most common form of logistic regression and is used for binary classification problems.
    Multinomial Logistic Regression: This is used when the dependent variable has three or more possible categories that are not ordered. For example, classifying animals into categories like "cat," "dog" or "sheep." It extends the binary logistic regression to handle multiple classes.
    Ordinal Logistic Regression: This type applies when the dependent variable has three or more categories with a natural order or ranking. Examples include ratings like "low," "medium" and "high." It takes the order of the categories into account when modeling.

Assumptions of Logistic Regression

Understanding the assumptions behind logistic regression is important to ensure the model is applied correctly, main assumptions are:

    Independent observations: Each data point is assumed to be independent of the others means there should be no correlation or dependence between the input samples.
    Binary dependent variables: It takes the assumption that the dependent variable must be binary, means it can take only two values. For more than two categories SoftMax functions are used.
    Linearity relationship between independent variables and log odds: The model assumes a linear relationship between the independent variables and the log odds of the dependent variable which means the predictors affect the log odds in a linear way.
    No outliers: The dataset should not contain extreme outliers as they can distort the estimation of the logistic regression coefficients.
    Large sample size: It requires a sufficiently large sample size to produce reliable and stable results.

Understanding Sigmoid Function

1. The sigmoid function is a important part of logistic regression which is used to convert the raw output of the model into a probability value between 0 and 1.

2. This function takes any real number and maps it into the range 0 to 1 forming an "S" shaped curve called the sigmoid curve or logistic curve. Because probabilities must lie between 0 and 1, the sigmoid function is perfect for this purpose.

3. In logistic regression, we use a threshold value usually 0.5 to decide the class label.

    If the sigmoid output is same or above the threshold, the input is classified as Class 1.
    If it is below the threshold, the input is classified as Class 0.

This approach helps to transform continuous input values into meaningful class predictions.
'''

result = chain.invoke({'text':text})
print(result)
