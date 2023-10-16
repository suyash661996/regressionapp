import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    st.title("Regression Model Builder")

    # Upload CSV file
    file = st.file_uploader("Upload CSV", type="csv")

    if file is not None:
        # Read CSV file
        data = pd.read_csv(file)

        st.subheader("Data Preview")
        st.write(data.head())

        # Clear NaN values
        data.dropna(inplace=True)

        # Select target variable
        target_variable = st.selectbox("Select the target variable", data.columns)

        # Select feature variables
        feature_variables = st.multiselect("Select the feature variables", data.columns)

        # Model selection
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso()
        }
        selected_model = st.sidebar.selectbox("Select Regression Model", list(models.keys()))

        if st.button("Build Model"):
            # Prepare X and y
            X = data[feature_variables]
            y = data[target_variable]

            # Model comparison
            best_model = None
            best_performance = float('-inf')

            for model_name, model in models.items():
                # Build the regression model
                model.fit(X, y)

                # Make predictions
                y_pred = model.predict(X)

                # Calculate performance parameters
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)

                st.subheader(f"{model_name} Summary")
                st.write("Intercept:", model.intercept_)
                st.write("Coefficients:", model.coef_)
                st.write("R-squared:", r2)
                st.write("Mean Squared Error:", mse)
                st.write("Mean Absolute Error:", mae)

                # Track best performing model
                if r2 > best_performance:
                    best_performance = r2
                    best_model = model_name

            st.subheader("Best Performing Model")
            st.write(best_model)

if __name__ == "__main__":
    main()
