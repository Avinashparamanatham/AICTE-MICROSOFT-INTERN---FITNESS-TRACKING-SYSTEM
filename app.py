import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

class FitnessTrackingApp:
    def __init__(self):
        # Load pre-trained models and scaler
        self.load_models()

    def load_models(self):
        """
        Load pre-trained models and scaler
        """
        try:
            with open('calories_model.pkl', 'rb') as f:
                self.calories_model = pickle.load(f)
            
            with open('heart_rate_model.pkl', 'rb') as f:
                self.heart_rate_model = pickle.load(f)
            
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.error("Please run the dataset processing script first to train and save models.")
            self.calories_model = None
            self.heart_rate_model = None
            self.scaler = None

    def run(self):
        """
        Main Streamlit app
        """
        # Page configuration
        st.set_page_config(
            page_title="🏋️ Fitness Tracking System", 
            page_icon="💪", 
            layout="wide"
        )

        # Sidebar navigation
        st.sidebar.title("🌟 Fitness Tracker")
        menu_options = [
            "🏠 Home Dashboard", 
            "📊 Fitness Analytics", 
            "🔮 Prediction Center", 
            "📈 Performance Tracker", 
            "🏆 Goal Setting"
        ]
        selected_menu = st.sidebar.radio("Navigate", menu_options)

        # Routing based on selected menu
        if selected_menu == "🏠 Home Dashboard":
            self.home_dashboard()
        elif selected_menu == "📊 Fitness Analytics":
            self.fitness_analytics()
        elif selected_menu == "🔮 Prediction Center":
            self.prediction_center()
        elif selected_menu == "📈 Performance Tracker":
            self.performance_tracker()
        else:
            self.goal_setting_page()

    def home_dashboard(self):
        """
        Home dashboard with key metrics
        """
        st.title("🏋️ Fitness Tracking Dashboard")
        
        # Placeholder for actual metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Users", 100, "🔼")
        with col2:
            st.metric("🔥 Avg Calories Burned", 450, "📈")
        with col3:
            st.metric("💓 Avg Heart Rate", 120, "❤️")
        with col4:
            st.metric("⏱️ Avg Exercise Duration", 45, "🚀")

    def fitness_analytics(self):
        """
        Fitness data analytics
        """
        st.title("📊 Fitness Analytics")
        
        # Generate sample data for demonstration
        np.random.seed(42)
        df = pd.DataFrame({
            'Age': np.random.randint(18, 65, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Calories': np.random.randint(200, 800, 100),
            'Heart_Rate': np.random.randint(60, 180, 100)
        })

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 Calories Distribution")
            fig_calories = px.histogram(
                df, x="Calories", color="Gender", 
                title="Calories Burned Distribution"
            )
            st.plotly_chart(fig_calories)
        
        with col2:
            st.subheader("💓 Heart Rate Analysis")
            fig_heart_rate = px.box(
                df, x="Gender", y="Heart_Rate", 
                title="Heart Rate by Gender"
            )
            st.plotly_chart(fig_heart_rate)

    def prediction_center(self):
        """
        Fitness metrics prediction page
        """
        st.title("🔮 Fitness Prediction Center")
        
        if not self.calories_model or not self.heart_rate_model:
            st.error("Models not loaded. Please train models first.")
            return

        # Prediction input form
        with st.form("fitness_prediction"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("🎂 Age", min_value=18, max_value=70, value=30)
                height = st.number_input("📏 Height (cm)", min_value=140, max_value=220, value=170)
                gender = st.selectbox("👥 Gender", ["Male", "Female"])
            
            with col2:
                weight = st.number_input("⚖️ Weight (kg)", min_value=40, max_value=150, value=70)
                duration = st.number_input("⏱️ Exercise Duration (mins)", min_value=10, max_value=180, value=45)
            
            predict_button = st.form_submit_button("🚀 Predict My Fitness Metrics")
        
        if predict_button:
            # Prepare input data
            input_data = pd.DataFrame({
                'Age': [age], 'Height': [height], 'Weight': [weight],
                'Duration': [duration], 'Gender': [gender],
                'Heart_Rate': [0], 'Body_Temp': [37.0]
            })
            
            # One-hot encode gender
            input_encoded = pd.get_dummies(input_data, columns=['Gender'], prefix='Gender')
            
            # Ensure all columns are present
            required_columns = [
                'Age', 'Height', 'Weight', 'Duration', 
                'Heart_Rate', 'Body_Temp', 
                'Gender_Male', 'Gender_Female'
            ]
            
            for col in required_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Select and order columns
            input_encoded = input_encoded[required_columns]
            
            # Scale input
            input_scaled = self.scaler.transform(input_encoded)
            
            # Predict
            calories_pred = self.calories_model.predict(input_scaled)[0]
            hr_pred = self.heart_rate_model.predict(input_scaled)[0]
            
            # Display predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🔥 Predicted Calories Burned", 
                          f"{calories_pred:.2f} cal", 
                          delta_color="off")
            
            with col2:
                st.metric("💓 Predicted Heart Rate", 
                          f"{hr_pred:.0f} bpm", 
                          delta_color="off")

    def performance_tracker(self):
        """
        Track user's fitness performance over time
        """
        st.title("📈 Performance Tracker")
        
        # Simulated performance data
        performance_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Calories Burned': np.cumsum(np.random.randint(200, 500, 10)),
            'Exercise Duration': np.cumsum(np.random.randint(30, 90, 10)),
            'Heart Rate': np.random.randint(70, 150, 10)
        })

        # Line chart for performance metrics
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_data['Date'], 
            y=performance_data['Calories Burned'], 
            mode='lines+markers', 
            name='Calories Burned'
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['Date'], 
            y=performance_data['Exercise Duration'], 
            mode='lines+markers', 
            name='Exercise Duration'
        ))
        
        fig.update_layout(
            title='Performance Metrics Over Time',
            xaxis_title='Date',
            yaxis_title='Metrics'
        )
        
        st.plotly_chart(fig)

    def goal_setting_page(self):
        """
        Goal setting and tracking page
        """
        st.title("🏆 Goal Setting")
        
        with st.form("fitness_goals"):
            st.subheader("Set Your Fitness Goals")
            
            goal_type = st.selectbox("Goal Type", [
                "Weight Loss", 
                "Muscle Gain", 
                "Endurance", 
                "Cardiovascular Health"
            ])
            
            target_value = st.number_input(
                "Target Value", 
                min_value=1, 
                max_value=1000, 
                value=10
            )
            
            time_frame = st.select_slider(
                "Time Frame", 
                options=[
                    "1 Month", 
                    "3 Months", 
                    "6 Months", 
                    "1 Year"
                ]
            )
            
            submitted = st.form_submit_button("Set Goal")
            
            if submitted:
                st.success(f"Goal Set: {goal_type} - {target_value} in {time_frame}")
                st.info("Track your progress regularly and adjust your plan as needed.")

def main():
    """
    Main function to run the Streamlit app
    """
    app = FitnessTrackingApp()
    app.run()

if __name__ == "__main__":
    main()