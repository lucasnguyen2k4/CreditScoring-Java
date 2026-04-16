import pandas as pd
import numpy as np

def generate_sample_data(n_records: int = 100) -> pd.DataFrame:
    """
    Generate professional credit scoring sample data.
    """
    data = []

    for i in range(1, n_records + 1):
        customer_id = f"C{i:03d}"
        
        # Personal info
        age = np.random.randint(22, 61)
        gender = np.random.choice(["Male", "Female"])
        marital_status = np.random.choice(["Single", "Married", "Divorced", "Widowed"], p=[0.35, 0.45, 0.15, 0.05])
        education = np.random.choice(["High School", "Undergraduate", "Graduate", "Postgraduate"], p=[0.25, 0.25, 0.35, 0.15])
        dependents = np.random.choice([0, 1, 2, 3], p=[0.3, 0.2, 0.35, 0.15])
        region = np.random.choice(["Urban", "Suburban", "Rural"], p=[0.5, 0.35, 0.15])
        
        # Employment
        if age < 25:
            employment_status = np.random.choice(["Employed", "Unemployed"], p=[0.7, 0.3])
        elif age > 58:
            employment_status = np.random.choice(["Employed", "Retired"], p=[0.6, 0.4])
        else:
            employment_status = np.random.choice(["Employed", "Self-employed", "Unemployed"], p=[0.80, 0.15, 0.05])
        
        if employment_status == "Employed":
            years_employed = min(age - 20, np.random.randint(1, 30))
            annual_income = np.random.randint(30000, 150000)
            employment_type = np.random.choice(["Full-time", "Part-time"], p=[0.85, 0.15])
        elif employment_status == "Self-employed":
            years_employed = np.random.randint(3, 30)
            annual_income = np.random.randint(50000, 160000)
            employment_type = "Self-employed"
        elif employment_status == "Retired":
            years_employed = 0
            annual_income = np.random.randint(40000, 100000)
            employment_type = "Retired"
        else:  # Unemployed
            years_employed = 0
            annual_income = 0
            employment_type = "Unemployed"
        
        # Financial info
        if annual_income > 0:
            loan_amount = np.random.randint(20000, 250000)
            max_monthly_debt = max(600, int(annual_income * 0.08 / 12))  # At least 600
            monthly_debt = np.random.randint(500, max_monthly_debt)
            debt_to_income_ratio = round(monthly_debt / (annual_income / 12), 2) if annual_income > 0 else np.nan
        else:
            loan_amount = np.random.randint(20000, 50000)
            monthly_debt = np.random.randint(500, 1000)
            debt_to_income_ratio = np.nan
        
        loan_purpose = np.random.choice(["Home", "Car", "Business", "Personal", "Education", "Debt Consolidation"])
        
        # Banking history
        bank_account_years = min(age - 18, np.random.randint(0, 30))
        
        # Credit history
        credit_history_length = min(bank_account_years, np.random.randint(1, 32))
        num_credit_cards = np.random.randint(1, 9)
        credit_utilization = round(np.random.uniform(0.25, 0.95), 2)
        num_late_payments = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])
        num_accounts = np.random.randint(2, 11)
        previous_defaults = np.random.choice([0, 1], p=[0.85, 0.15])
        
        # Credit score
        base_score = 700
        if annual_income > 100000:
            base_score += 50
        elif annual_income < 40000:
            base_score -= 50
        
        base_score -= num_late_payments * 30
        base_score -= int(credit_utilization * 100)
        base_score -= previous_defaults * 100
        base_score += int(credit_history_length * 2)
        
        if employment_status == "Unemployed":
            base_score -= 100
        
        credit_score_external = max(495, min(850, base_score + np.random.randint(-30, 30)))
        
        # Savings and checking (with missing values for unemployed)
        if employment_status == "Unemployed" and np.random.random() < 0.8:
            savings_balance = np.nan
            checking_balance = np.nan
        else:
            savings_balance = np.random.randint(0, int(annual_income * 0.8)) if annual_income > 0 else np.random.randint(500, 2000)
            checking_balance = np.random.randint(0, int(annual_income * 0.15)) if annual_income > 0 else np.random.randint(200, 1000)
        
        home_ownership = np.random.choice(["Own", "Rent", "Mortgage"], p=[0.4, 0.35, 0.25])
        
        # Target variable (default)
        default_prob = 0.1  # Base probability
        
        if credit_utilization > 0.75:
            default_prob += 0.3
        if num_late_payments >= 3:
            default_prob += 0.3
        if credit_score_external < 600:
            default_prob += 0.25
        if employment_status == "Unemployed":
            default_prob += 0.35
        if previous_defaults > 0:
            default_prob += 0.2
        if debt_to_income_ratio > 0.5 if not pd.isna(debt_to_income_ratio) else False:
            default_prob += 0.15
        
        default_prob = min(default_prob, 0.95)
        default = 1 if np.random.random() < default_prob else 0
        
        # Append record
        data.append({
            'customer_id': customer_id,
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education': education,
            'employment_status': employment_status,
            'years_employed': years_employed,
            'annual_income': annual_income,
            'monthly_debt': monthly_debt,
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'credit_history_length': credit_history_length,
            'num_credit_cards': num_credit_cards,
            'credit_utilization': credit_utilization,
            'num_late_payments': num_late_payments,
            'num_accounts': num_accounts,
            'home_ownership': home_ownership,
            'dependents': dependents,
            'region': region,
            'bank_account_years': bank_account_years,
            'previous_defaults': previous_defaults,
            'debt_to_income_ratio': debt_to_income_ratio,
            'savings_balance': savings_balance,
            'checking_balance': checking_balance,
            'employment_type': employment_type,
            'credit_score_external': credit_score_external,
            'default': default
        })

    return pd.DataFrame(data)
