# Build and deploy machine learning web app using TPOT and Streamlit

This app uses the python library TPOT to build optimized machine learning models with minimal lines of code.  It is designed for productivity and continuous integration, continous development.

The model is then deployed with Streamlit, a library in Python that allows for simpler deployment of an interactive frontend space for the user.

The machine learning algorithm used is a catogical boost ensemble algorithm.  All 7 features were used.  Label encoding on Sex, smoker, and region columns.  

# Conclusions
1. The most important features when determining healthcare costs per individual were Age, Sex, and number of children.
2. The factors BMI, Smoker/Nonsmoker, and region appeared to have negligible if any affects on the total costs, according to the model.  This is a potential red flag the model needs future improvement.
3.  More data needs to be acquired as well as possible weights added to features such as Smoker, and BMI.  Factors that any expert would agree ought to significatly affect the results.
4.  The model is a prototype


# Limitations
1. The dataset is small.  The sample population for training may not accurately reflect the true distribution of each variable.

See IPython Notebook for details

