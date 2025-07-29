import cohere

co = cohere.Client("")  #replace w ur key

def generate_health_tips(input_data, prediction):
    labels = [
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
        "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
    ]

    formatted_data = "\n".join(
        f"{label}: {value}" for label, value in zip(labels, input_data[0])
    )

    condition = "have diabetes" if prediction == 1 else "do not have diabetes"

    prompt = f"""
    A patient entered the following medical information:
    {formatted_data}

    The model predicts the patient may {condition}.

    Based on this, provide personalized, medically safe, non-alarming, and practical health tips about:
    - Blood sugar control
    - Lifestyle improvements
    - Diet and exercise
    - Preventing future complications

    Give 5-6 short, clear, medically safe tips for lifestyle and health improvement.
    Keep the tips very brief, practical, and non-alarming.
    """

    try:
        response = co.chat(
            model="command-r", 
            message=prompt,
            temperature=0.7
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error generating tips: {e}"
