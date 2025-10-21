#!/usr/bin/env python3
"""
Bank Statement Visualization Tool - Assignment 1.2
Interactive tool to view monthly spend reports (August 2020 - January 2021)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class BankStatementVisualizer:
    def __init__(self, data_file: str = 'extracted_transactions.json'):
        """Initialize the visualizer with transaction data"""
        self.data_file = data_file
        self.transactions = self.load_data()
        self.df = pd.DataFrame(self.transactions)
        self.prepare_data()
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load transaction data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} transactions from {self.data_file}")
            return data
        except FileNotFoundError:
            print(f"Error: {self.data_file} not found!")
            return []
    
    def prepare_data(self):
        """Prepare data for visualization"""
        if self.df.empty:
            print("No data to prepare")
            return
            
        # Convert date columns to datetime
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
        self.df['transaction_postdate'] = pd.to_datetime(self.df['transaction_postdate'])
        
        # Extract month and year for filtering
        self.df['year_month'] = self.df['transaction_date'].dt.to_period('M')
        self.df['month_name'] = self.df['transaction_date'].dt.strftime('%B %Y')
        
        # Filter out payments for spending analysis
        self.spending_df = self.df[self.df['spend_category'] != 'PAYMENT'].copy()
        
        print("Data prepared successfully!")
        print(f"Available months: {sorted(self.df['year_month'].unique())}")
    
    def get_available_months(self) -> List[str]:
        """Get list of available months in the data"""
        if self.df.empty:
            return []
        months = sorted(self.df['year_month'].unique())
        return [str(month) for month in months]
    
    def get_monthly_data(self, month: str) -> pd.DataFrame:
        """Get data for a specific month"""
        if self.df.empty:
            return pd.DataFrame()
        
        month_period = pd.Period(month)
        monthly_data = self.df[self.df['year_month'] == month_period].copy()
        return monthly_data
    
    def get_monthly_spending_data(self, month: str) -> pd.DataFrame:
        """Get spending data (excluding payments) for a specific month"""
        if self.spending_df.empty:
            return pd.DataFrame()
        
        month_period = pd.Period(month)
        monthly_spending = self.spending_df[self.spending_df['year_month'] == month_period].copy()
        return monthly_spending
    
    def calculate_monthly_summary(self, month: str) -> Dict[str, Any]:
        """Calculate summary statistics for a month"""
        monthly_data = self.get_monthly_data(month)
        monthly_spending = self.get_monthly_spending_data(month)
        
        if monthly_data.empty:
            return {}
        
        summary = {
            'month': month,
            'total_transactions': len(monthly_data),
            'total_spending': monthly_spending['amount'].sum() if not monthly_spending.empty else 0,
            'total_payments': monthly_data[monthly_data['spend_category'] == 'PAYMENT']['amount'].sum(),
            'average_transaction': monthly_spending['amount'].mean() if not monthly_spending.empty else 0,
            'max_transaction': monthly_spending['amount'].max() if not monthly_spending.empty else 0,
            'categories': monthly_spending['spend_category'].value_counts().to_dict() if not monthly_spending.empty else {},
            'top_merchant': monthly_spending['transaction_description'].value_counts().head(1).to_dict() if not monthly_spending.empty else {}
        }
        
        return summary
    
    def create_monthly_spend_chart(self, month: str):
        """Create a comprehensive spend report chart for a month"""
        monthly_spending = self.get_monthly_spending_data(month)
        
        if monthly_spending.empty:
            print(f"No spending data available for {month}")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Spend Report - {month}', fontsize=16, fontweight='bold')
        
        # 1. Category-wise spending (Pie Chart)
        category_totals = monthly_spending.groupby('spend_category')['amount'].sum()
        if not category_totals.empty:
            ax1.pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Spending by Category')
        
        # 2. Daily spending trend (Line Chart)
        daily_spending = monthly_spending.groupby(monthly_spending['transaction_date'].dt.day)['amount'].sum()
        ax2.plot(daily_spending.index, daily_spending.values, marker='o', linewidth=2, markersize=6)
        ax2.set_title('Daily Spending Trend')
        ax2.set_xlabel('Day of Month')
        ax2.set_ylabel('Amount ($)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Top merchants (Bar Chart)
        top_merchants = monthly_spending['transaction_description'].value_counts().head(8)
        if not top_merchants.empty:
            ax3.barh(range(len(top_merchants)), top_merchants.values)
            ax3.set_yticks(range(len(top_merchants)))
            ax3.set_yticklabels([desc[:30] + '...' if len(desc) > 30 else desc for desc in top_merchants.index])
            ax3.set_title('Top Merchants by Transaction Count')
            ax3.set_xlabel('Number of Transactions')
        
        # 4. Category spending amounts (Bar Chart)
        if not category_totals.empty:
            ax4.bar(range(len(category_totals)), category_totals.values)
            ax4.set_xticks(range(len(category_totals)))
            ax4.set_xticklabels(category_totals.index, rotation=45, ha='right')
            ax4.set_title('Spending Amount by Category')
            ax4.set_ylabel('Amount ($)')
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_table(self, month: str):
        """Create a summary table for a month"""
        summary = self.calculate_monthly_summary(month)
        
        if not summary:
            print(f"No data available for {month}")
            return
        
        print(f"\n{'='*60}")
        print(f"SPEND REPORT SUMMARY - {month}")
        print(f"{'='*60}")
        print(f"Total Transactions: {summary['total_transactions']}")
        print(f"Total Spending: ${summary['total_spending']:.2f}")
        print(f"Total Payments: ${summary['total_payments']:.2f}")
        print(f"Average Transaction: ${summary['average_transaction']:.2f}")
        print(f"Largest Transaction: ${summary['max_transaction']:.2f}")
        
        print(f"\nSpending by Category:")
        for category, amount in summary['categories'].items():
            print(f"  {category}: ${amount:.2f}")
        
        if summary['top_merchant']:
            merchant, count = list(summary['top_merchant'].items())[0]
            print(f"\nMost Frequent Merchant: {merchant} ({count} transactions)")
    
    def interactive_month_selector(self):
        """Interactive tool to select and view monthly reports"""
        available_months = self.get_available_months()
        
        if not available_months:
            print("No data available for visualization")
            return
        
        print(f"\n{'='*60}")
        print("BANK STATEMENT VISUALIZATION TOOL")
        print(f"{'='*60}")
        print("Available months for spend reports:")
        
        for i, month in enumerate(available_months, 1):
            print(f"{i}. {month}")
        
        print(f"{len(available_months) + 1}. View All Months Summary")
        print("0. Exit")
        
        while True:
            try:
                choice = input(f"\nSelect a month (0-{len(available_months) + 1}): ").strip()
                
                if choice == '0':
                    print("Goodbye!")
                    break
                elif choice == str(len(available_months) + 1):
                    self.create_all_months_summary()
                elif choice.isdigit() and 1 <= int(choice) <= len(available_months):
                    selected_month = available_months[int(choice) - 1]
                    self.view_monthly_report(selected_month)
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def view_monthly_report(self, month: str):
        """View comprehensive monthly report"""
        print(f"\nGenerating report for {month}...")
        
        # Show summary table
        self.create_summary_table(month)
        
        # Show charts
        self.create_monthly_spend_chart(month)
        
        # Show detailed transaction list
        monthly_data = self.get_monthly_data(month)
        if not monthly_data.empty:
            print(f"\nDetailed Transactions for {month}:")
            print("-" * 80)
            for _, transaction in monthly_data.iterrows():
                print(f"{transaction['transaction_date'].strftime('%Y-%m-%d')} | "
                      f"{transaction['transaction_description'][:40]:<40} | "
                      f"{transaction['spend_category']:<25} | "
                      f"${transaction['amount']:>8.2f}")
    
    def create_all_months_summary(self):
        """Create summary for all months"""
        available_months = self.get_available_months()
        
        print(f"\n{'='*80}")
        print("ALL MONTHS SUMMARY")
        print(f"{'='*80}")
        
        total_spending = 0
        total_transactions = 0
        
        for month in available_months:
            summary = self.calculate_monthly_summary(month)
            if summary:
                print(f"{month:<15} | Transactions: {summary['total_transactions']:<3} | "
                      f"Spending: ${summary['total_spending']:>8.2f} | "
                      f"Payments: ${summary['total_payments']:>8.2f}")
                total_spending += summary['total_spending']
                total_transactions += summary['total_transactions']
        
        print("-" * 80)
        print(f"{'TOTAL':<15} | Transactions: {total_transactions:<3} | "
              f"Spending: ${total_spending:>8.2f}")
        
        # Create overall trend chart
        self.create_trend_chart()
    
    def create_trend_chart(self):
        """Create trend chart across all months"""
        if self.spending_df.empty:
            print("No spending data available for trend analysis")
            return
        
        monthly_totals = self.spending_df.groupby('year_month')['amount'].sum()
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(monthly_totals)), monthly_totals.values, marker='o', linewidth=2, markersize=8)
        plt.title('Monthly Spending Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Total Spending ($)')
        plt.xticks(range(len(monthly_totals)), [str(month) for month in monthly_totals.index], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("Bank Statement Visualization Tool - Assignment 1.2")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = BankStatementVisualizer()
    
    if visualizer.df.empty:
        print("No transaction data found. Please ensure extracted_transactions.json exists.")
        return
    
    # Start interactive tool
    visualizer.interactive_month_selector()

if __name__ == "__main__":
    main()
