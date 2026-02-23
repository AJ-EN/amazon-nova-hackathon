from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")

# This stores submitted PA requests in memory during development
# In production you'd use a database, but for the hackathon this is fine
submitted_requests = []


@app.route("/")
def pa_form():
    """Renders the prior authorization request form."""
    return render_template("pa_form.html")


@app.route("/submit", methods=["POST"])
def submit_pa():
    """
    Receives a submitted PA form and stores it.
    Nova Act will POST to this endpoint after filling the form.
    """
    data = {
        "patient_name": request.form.get("patient_name"),
        "date_of_birth": request.form.get("date_of_birth"),
        "member_id": request.form.get("member_id"),
        "payer_name": request.form.get("payer_name"),
        "provider_npi": request.form.get("provider_npi"),
        "requested_service": request.form.get("requested_service"),
        "facility_name": request.form.get("facility_name"),
        "diagnosis_code": request.form.get("diagnosis_code"),
        "procedure_code": request.form.get("procedure_code"),
        "clinical_justification": request.form.get("clinical_justification"),
        "policy_id": request.form.get("policy_id"),
        "denial_risk_score": request.form.get("denial_risk_score"),
        "urgency": request.form.get("urgency"),
    }
    submitted_requests.append(data)
    print(
        f"✅ PA Request received: {data['patient_name']} - {data['procedure_code']}")
    return jsonify({"status": "submitted", "reference": f"PA-{len(submitted_requests):04d}"})


@app.route("/requests")
def view_requests():
    """A simple admin view to see all submitted PA requests — useful for demo."""
    return jsonify(submitted_requests)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "submitted_count": len(submitted_requests)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
