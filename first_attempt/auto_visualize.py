#!/usr/bin/env python3
"""
Simple visualization script that automatically displays charts
No user input required - just run and see the visualizations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the transaction data"""
    # Load data
    with open('extracted_transactions.json', 'r') as f:
        transactions = json.load(f)
    
    df = pd.DataFrame(transactions)
    
    # Convert dates
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['transaction_postdate'] = pd.to_datetime(df['transaction_postdate'])
    
    # Extract month and year
    df['year_month'] = df['transaction_date'].dt.to_period('M')
    df['month_name'] = df['transaction_date'].dt.strftime('%B %Y')
    
    # Filter spending data (exclude payments)
    spending_df = df[df['spend_category'] != 'PAYMENT'].copy()
    
    return df, spending_df

def create_overview_charts(df, spending_df):
    """Create overview charts for all data"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bank Statement Analysis - Overview', fontsize=16, fontweight='bold')
    
    # 1. Monthly spending trend
    monthly_spending = spending_df.groupby('year_month')['amount'].sum()
    ax1.plot(range(len(monthly_spending)), monthly_spending.values, 
             marker='o', linewidth=3, markersize=8, color='#2E86AB')
    ax1.set_title('Monthly Spending Trend', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Spending ($)')
    ax1.set_xticks(range(len(monthly_spending)))
    ax1.set_xticklabels([str(month) for month in monthly_spending.index], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Category distribution (pie chart)
    category_totals = spending_df.groupby('spend_category')['amount'].sum()
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_totals)))
    wedges, texts, autotexts = ax2.pie(category_totals.values, 
                                       labels=category_totals.index,
                                       autopct='%1.1f%%', 
                                       startangle=90,
                                       colors=colors)
    ax2.set_title('Spending by Category', fontsize=14, fontweight='bold')
    
    # 3. Top merchants
    top_merchants = spending_df['transaction_description'].value_counts().head(10)
    ax3.barh(range(len(top_merchants)), top_merchants.values, color='#A23B72')
    ax3.set_yticks(range(len(top_merchants)))
    ax3.set_yticklabels([desc[:25] + '...' if len(desc) > 25 else desc 
                        for desc in top_merchants.index])
    ax3.set_title('Top 10 Merchants by Transaction Count', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Transactions')
    
    # 4. Category spending amounts
    ax4.bar(range(len(category_totals)), category_totals.values, color='#F18F01')
    ax4.set_xticks(range(len(category_totals)))
    ax4.set_xticklabels(category_totals.index, rotation=45, ha='right')
    ax4.set_title('Total Spending by Category', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Amount ($)')
    
    plt.tight_layout()
    plt.show()

def create_monthly_breakdown(spending_df):
    """Create detailed monthly breakdown"""
    
    # Get unique months
    months = sorted(spending_df['year_month'].unique())
    
    # Create subplots for each month
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Monthly Spending Breakdown', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, month in enumerate(months):
        if i >= 6:  # Limit to 6 subplots
            break
            
        monthly_data = spending_df[spending_df['year_month'] == month]
        
        if not monthly_data.empty:
            # Category breakdown for this month
            category_totals = monthly_data.groupby('spend_category')['amount'].sum()
            
            axes[i].pie(category_totals.values, 
                       labels=category_totals.index,
                       autopct='%1.1f%%', 
                       startangle=90)
            axes[i].set_title(f'{month}', fontsize=12, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{month}', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for i in range(len(months), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def create_daily_patterns(spending_df):
    """Create daily spending patterns"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Daily Spending Patterns', fontsize=16, fontweight='bold')
    
    # Daily spending trend
    daily_spending = spending_df.groupby(spending_df['transaction_date'].dt.day)['amount'].sum()
    ax1.plot(daily_spending.index, daily_spending.values, 
             marker='o', linewidth=2, markersize=6, color='#C73E1D')
    ax1.set_title('Average Daily Spending by Day of Month', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Day of Month')
    ax1.set_ylabel('Total Amount ($)')
    ax1.grid(True, alpha=0.3)
    
    # Day of week patterns
    spending_df['day_of_week'] = spending_df['transaction_date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_totals = spending_df.groupby('day_of_week')['amount'].sum().reindex(day_order)
    
    ax2.bar(range(len(daily_totals)), daily_totals.values, color='#3B1F2B')
    ax2.set_xticks(range(len(daily_totals)))
    ax2.set_xticklabels(daily_totals.index, rotation=45)
    ax2.set_title('Spending by Day of Week', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Amount ($)')
    
    plt.tight_layout()
    plt.show()

def print_summary_stats(df, spending_df):
    """Print summary statistics"""
    
    print("="*80)
    print("BANK STATEMENT ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Total Transactions: {len(df)}")
    print(f"Total Spending: ${spending_df['amount'].sum():.2f}")
    print(f"Average Transaction: ${spending_df['amount'].mean():.2f}")
    print(f"Largest Transaction: ${spending_df['amount'].max():.2f}")
    print(f"Smallest Transaction: ${spending_df['amount'].min():.2f}")
    
    print(f"\nDate Range: {df['transaction_date'].min().strftime('%Y-%m-%d')} to {df['transaction_date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nTop Categories:")
    category_totals = spending_df.groupby('spend_category')['amount'].sum().sort_values(ascending=False)
    for category, amount in category_totals.head(5).items():
        print(f"  {category}: ${amount:.2f}")
    
    print(f"\nMost Frequent Merchants:")
    top_merchants = spending_df['transaction_description'].value_counts().head(5)
    for merchant, count in top_merchants.items():
        print(f"  {merchant}: {count} transactions")

def main():
    """Main function to run all visualizations"""
    
    print("Loading and preparing data...")
    df, spending_df = load_and_prepare_data()
    
    print("Creating visualizations...")
    print("(Charts will appear in separate windows)")
    
    # Print summary statistics
    print_summary_stats(df, spending_df)
    
    # Create visualizations
    print("\nGenerating Overview Charts...")
    create_overview_charts(df, spending_df)
    
    print("Generating Monthly Breakdown...")
    create_monthly_breakdown(spending_df)
    
    print("Generating Daily Patterns...")
    create_daily_patterns(spending_df)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("All charts have been displayed.")
    print("="*80)

if __name__ == "__main__":
    main()
