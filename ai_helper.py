import openai

openai.api_key = ""

def generate_health_tips(input_data, prediction):
    print("woking")
    labels = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
              "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]
    formatted_data = "\n".join(
        f"{label}: {value}" for label, value in zip(labels, input_data[0])
    )

    condition = "have diabetes" if prediction == 1 else "do not have diabetes"

    prompt = f"""
    A patient entered the following medical data:
    {formatted_data}

    The model predicts the patient may {condition}.

    Based on this information, give personalized, practical, and safe health tips to manage glucose levels, lifestyle, and general wellness. Keep it simple, non-alarming, and medically reasonable.
    """

    try:
        response = openai.ChatCompletion.create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating tips: {e}"

