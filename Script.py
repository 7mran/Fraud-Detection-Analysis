import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def load_data():
    """Load the three data files."""
    acct_info = pd.read_csv("Step Up - Experian Account Data.csv")
    holder_info = pd.read_csv("Step Up - Experian Acount Holder Data.csv")
    fraud_indicators = pd.read_csv("Mule Flag.csv")
    return acct_info, holder_info, fraud_indicators


def audit_data(df, name):
    """Perform high-level data audit on a dataframe."""
    print(f"\n{name} Data Columns:")
    print("\n".join(df.columns.tolist()))
    print(f"\n{name} Data Length: {len(df)} records")

    # Check for duplicates
    duplicates = df[df.duplicated()]
    print(f"{name} Data Duplicates: {len(duplicates)}")

    # Calculate statistics for each column
    stats_dict = {}
    for column in df.columns:
        val_counts = df[column].value_counts(dropna=False)
        missing_count = df[column].isna().sum()
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0

        stats_dict[column] = {
            "value_counts": val_counts,
            "%_blank": missing_pct
        }

    return stats_dict


def clean_data(acct_info, holder_info, fraud_indicators):
    """Clean and prepare the data for analysis."""
    # Remove duplicate rows
    holder_info = holder_info.drop_duplicates()

    # Fill missing values with sensible defaults
    acct_default_vals = {
        "AccountLength": -1,
        "AverageBalance": -1,
        "NumTransactions": -1,
        "NumDeposits": -1,
        "NumWithdrawals": -1,
        "NumTransfers": -1,
        "NumLoans": -1,
        "NumCreditCards": -1,
        "NumSavingsAccounts": -1,
    }
    acct_info = acct_info.fillna(value=acct_default_vals)

    holder_default_vals = {
        'DateOfBirth': 'Missing',
        'Gender': 'Missing',
        'Income': -1,
        'CreditScore': -1,
        'LoanAmount': -1,
        'EmploymentStatus': 'Missing',
        'MaritalStatus': 'Missing',
        'OccupancyStatus': 'Missing',
        'NumDependents': -1,
        'SocialMediaUsageHours': -1,
        'ShoppingFrequencyPerMonth': -1,
        'HealthInsuranceStatus': 'Missing'
    }
    holder_info = holder_info.fillna(value=holder_default_vals)

    # Fill fraud indicators with 0 (assuming non-flagged accounts)
    fraud_indicators = fraud_indicators.fillna(0)

    return acct_info, holder_info, fraud_indicators


def merge_datasets(acct_info, holder_info, fraud_indicators):
    """Combine the three dataframes into one."""
    combined_data = acct_info.merge(holder_info, how='left', on='Identifier')
    combined_data = combined_data.merge(fraud_indicators, how='left', on='Identifier')
    return combined_data


def create_derived_features(df):
    """Create new features from existing data."""
    current_date = datetime.today()

    # Create age column
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], format='%d/%m/%Y', errors='coerce')
    df['Age'] = df['DateOfBirth'].apply(
        lambda birth_date: current_date.year - birth_date.year -
                           ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))
    )

    # Create age categories
    age_ranges = [0, 17, 24, 35, 45, 60, 100]
    age_categories = ['0-17', '18-24', '25-35', '36-45', '46-60', '60+']
    df['AgeCategory'] = pd.cut(df['Age'], bins=age_ranges, labels=age_categories, right=False)

    # Create income categories
    income_ranges = [0, 10000, 20000, 30000, 40000, 60000, 80000, 100000]
    income_categories = ['0-10k', '10k-20k', '20k-30k', '30k-40k', '40k-60k', '60k-80k', '80k+']
    df['IncomeCategory'] = pd.cut(df['Income'], bins=income_ranges, labels=income_categories, right=False)

    return df


def analyze_fraud_patterns(df):
    """Analyze patterns in mule account data."""
    # Group by age category
    age_cat_results = df.groupby('AgeCategory')['MuleAccount'].sum()
    print("Mule accounts by age category:")
    print(age_cat_results)

    # Group by gender
    gender_breakdown = df.groupby('Gender')['MuleAccount'].sum()
    print("\nMule accounts by gender:")
    print(gender_breakdown)

    # Analyze combinations of characteristics
    profile_features = ['AgeCategory', 'Gender']
    fraud_analysis = df.groupby(profile_features)['MuleAccount'].sum().reset_index()
    fraud_analysis = fraud_analysis.sort_values(by='MuleAccount', ascending=False)

    print("\nCharacteristics with the highest number of mule accounts:")
    print(fraud_analysis.head(5))

    return age_cat_results, gender_breakdown, fraud_analysis


def create_visualizations(age_cat_results, gender_breakdown):
    """Create visualizations for the analysis."""
    # Age group chart
    plt.figure(figsize=(10, 5))
    age_cat_results.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Number of Mule Accounts by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Mule Accounts')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Gender chart
    plt.figure(figsize=(6, 4))
    gender_breakdown.plot(kind='bar', color='purple', edgecolor='black')
    plt.title('Number of Mule Accounts by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Number of Mule Accounts')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Load data
    acct_info, holder_info, fraud_indicators = load_data()

    # Data audit
    acct_stats = audit_data(acct_info, "Account")
    holder_stats = audit_data(holder_info, "Account Holder")
    fraud_stats = audit_data(fraud_indicators, "Mule Flag")

    # Verify identifier consistency across datasets
    ids_match = (set(acct_info['Identifier']) == set(holder_info['Identifier']) ==
                 set(fraud_indicators['Identifier']))
    unique_ids = len(list(acct_info['Identifier'])) == len(set(acct_info['Identifier']))
    print(f"\nIdentifier consistency: {ids_match}")
    print(f"No duplicate identifiers: {unique_ids}")

    # Clean data
    acct_info, holder_info, fraud_indicators = clean_data(acct_info, holder_info, fraud_indicators)

    # Merge datasets
    combined_data = merge_datasets(acct_info, holder_info, fraud_indicators)

    # Create derived features
    combined_data = create_derived_features(combined_data)

    # Analyze fraud patterns
    age_results, gender_results, fraud_analysis = analyze_fraud_patterns(combined_data)

    # Create visualizations
    create_visualizations(age_results, gender_results)

    return combined_data


if __name__ == "__main__":
    final_data = main()