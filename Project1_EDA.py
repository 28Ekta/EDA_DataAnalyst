"""Interactive EDA application for the retail customer sales project.
Run: streamlit run Project1_EDA.py
"""
from __future__ import annotations
import io, os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Let users run this file from VS Code or PowerShell without invoking Streamlit manually.
if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    import subprocess
    import sys
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]))

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Retail Sales EDA", page_icon="bar_chart", layout="wide")
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "retail_customer_sales_dataset.csv"
NUMERIC = ["Age", "Income", "Quantity", "Unit_Price", "Discount", "Sales", "Cost", "Profit"]
CATS = ["Region", "Product_Category", "Customer_Segment"]

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes | None, path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes)) if file_bytes is not None else pd.read_csv(path)

def standardize(series: pd.Series, aliases: dict[str,str] | None = None) -> pd.Series:
    s=series.astype("string").str.strip()
    s=s.str.replace(r"\s+", " ", regex=True)
    if aliases:
        return s.str.casefold().map({**{x.casefold():y for x,y in aliases.items()}, **{y.casefold():y for y in aliases.values()}}).fillna(s.str.title())
    return s.str.title()

def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    duplicate_flags = raw.duplicated(keep="first").to_numpy()
    d=raw.copy()
    d["_Source_Row"] = np.arange(2, len(d)+2)
    d["Flag_Duplicate"] = duplicate_flags
    # Parse dates with coercion so invalid formats remain auditable as NaT.
    for c in ["Order_Date","Delivery_Date"]:
        d[c+"_Raw"] = d[c].astype("string")
        d[c] = pd.to_datetime(d[c], errors="coerce", format="mixed")
    gender={"m":"Male","male":"Male","f":"Female","female":"Female"}
    d["Gender"] = standardize(d["Gender"], gender)
    region={"nort":"North","north":"North","south":"South","sth":"South","east":"East","west":"West"}
    d["Region"] = standardize(d["Region"], region)
    for c in ["Customer_Segment","Product_Category","Product","State","City","Payment_Method","Delivery_Mode","Satisfaction_Level","Returned"]:
        if c in d: d[c]=d[c].astype("string").str.strip().str.replace(r"\s+"," ",regex=True)
    for c in NUMERIC+["Customer_Rating"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    # Invalid values are retained in source, set missing in analysis copy, and disclosed via flags.
    d["Flag_Invalid_Age"] = d["Age"].notna() & ~d["Age"].between(0,120)
    d.loc[d["Flag_Invalid_Age"],"Age"] = np.nan
    d["Flag_Invalid_Discount"] = d["Discount"].notna() & ~d["Discount"].between(0,0.5)
    d.loc[d["Flag_Invalid_Discount"],"Discount"] = np.nan
    d["Flag_Invalid_Rating"] = d["Customer_Rating"].notna() & ~d["Customer_Rating"].between(1,5)
    d.loc[d["Flag_Invalid_Rating"],"Customer_Rating"] = np.nan
    d["Flag_Invalid_Income"] = d["Income"].notna() & (d["Income"] < 0)
    d.loc[d["Flag_Invalid_Income"],"Income"] = np.nan
    d["Flag_Invalid_Quantity"] = d["Quantity"].notna() & (d["Quantity"] <= 0)
    d.loc[d["Flag_Invalid_Quantity"],"Quantity"] = np.nan
    d["Flag_Invalid_Unit_Price"] = d["Unit_Price"].notna() & (d["Unit_Price"] <= 0)
    d.loc[d["Flag_Invalid_Unit_Price"],"Unit_Price"] = np.nan
    d["Flag_Negative_Sales"] = d["Sales"].notna() & (d["Sales"] < 0)
    d.loc[d["Flag_Negative_Sales"],"Sales"] = np.nan
    d["Flag_Date_Order"] = d["Order_Date"].isna()
    d["Flag_Date_Delivery"] = d["Delivery_Date"].isna()
    d["Flag_Date_Format_Order"] = d["Order_Date_Raw"].notna() & ~d["Order_Date_Raw"].str.strip().str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    d["Flag_Date_Format_Delivery"] = d["Delivery_Date_Raw"].notna() & ~d["Delivery_Date_Raw"].str.strip().str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    d["Flag_Delivery_Before_Order"] = d["Delivery_Date"].notna() & d["Order_Date"].notna() & (d["Delivery_Date"]<d["Order_Date"])
    d.loc[d["Flag_Delivery_Before_Order"],"Delivery_Date"] = pd.NaT
    d["Profit_Margin"] = np.where(d["Sales"]!=0, d["Profit"]/d["Sales"], np.nan)
    d["Delivery_Days"] = (d["Delivery_Date"]-d["Order_Date"]).dt.days
    customer_totals = d.loc[~d["Flag_Duplicate"]].groupby("Customer_ID")["Sales"].sum()
    d["Revenue_per_Customer"] = d["Customer_ID"].map(customer_totals)
    return d, d.loc[d["Flag_Duplicate"]].copy()

def iqr_flags(s: pd.Series) -> pd.Series:
    q1,q3=s.quantile(.25),s.quantile(.75); spread=q3-q1
    return s.notna() & ((s<q1-1.5*spread)|(s>q3+1.5*spread))

st.title("Retail Customer Sales | Exploratory Data Analysis")
st.caption("A two-year look at customer orders, data quality, and sales patterns.")
with st.sidebar:
    st.header("Data and filters")
    uploaded=st.file_uploader("Use another CSV (optional)", type="csv")
    if uploaded:
        raw=load_csv(uploaded.getvalue(), str(DEFAULT_DATA))
    elif DEFAULT_DATA.exists():
        raw=load_csv(None,str(DEFAULT_DATA))
    else:
        st.error("Place retail_customer_sales_dataset.csv next to this app."); st.stop()
    required_columns = ["Transaction_ID", "Customer_ID", "Order_Date", "Delivery_Date", "Age", "Gender", "Income", "Region", "Customer_Segment", "Product_Category", "Product", "Quantity", "Unit_Price", "Discount", "Sales", "Cost", "Profit", "Payment_Method", "Delivery_Mode", "Customer_Rating", "Satisfaction_Level", "Returned"]
    missing_columns = [column for column in required_columns if column not in raw.columns]
    if missing_columns:
        st.error("This CSV is missing required columns: " + ", ".join(missing_columns))
        st.stop()
    cleaned, duplicates=clean_data(raw)
    region_opts=sorted(cleaned.Region.dropna().unique())
    selected=st.multiselect("Region",region_opts,default=region_opts)
    date_range=st.date_input("Order date range", value=(cleaned.Order_Date.min().date(),cleaned.Order_Date.max().date()))
    show_duplicates=st.checkbox("Include duplicate records in views",False)
base=cleaned if show_duplicates else cleaned.loc[~cleaned.Flag_Duplicate]
if selected:
    base=base.loc[base.Region.isin(selected)]
else:
    base=base.iloc[0:0]
if isinstance(date_range,tuple) and len(date_range)==2:
    base=base.loc[base.Order_Date.dt.date.between(date_range[0],date_range[1])]

st.metric("Transactions in view",f"{len(base):,}")
tabs=st.tabs(["Overview", "Data quality", "Numbers", "Business questions", "Cleaning notes"])
with tabs[0]:
    a,b,c,d=st.columns(4); a.metric("Source rows", f"{len(raw):,}"); b.metric("Exact duplicates", f"{len(duplicates):,}"); c.metric("Sales in view", f"INR {base.Sales.sum():,.0f}"); d.metric("Customers in view", f"{base.Customer_ID.nunique():,}")
    st.subheader("Dataset structure")
    st.write(f"{raw.shape[0]:,} rows by {raw.shape[1]} source columns. Date coverage: {cleaned.Order_Date.min():%d %b %Y} to {cleaned.Order_Date.max():%d %b %Y}.")
    st.dataframe(pd.DataFrame({"Column":raw.columns,"Source type":raw.dtypes.astype(str).values,"Missing":raw.isna().sum().values,"Missing %":(100*raw.isna().mean()).round(2).values}),hide_index=True,width="stretch")
    if not base.empty:
        median_order = base.Sales.median()
        mean_order = base.Sales.mean()
        high_region = base.groupby("Region").Sales.mean().idxmax()
        high_region_value = base.groupby("Region").Sales.mean().max()
        st.markdown(f"The typical order in this view is **INR {median_order:,.0f}** (median). The average is **INR {mean_order:,.0f}**, which is higher because a smaller number of large orders pull it upward.")
        st.markdown(f"**{high_region}** has the highest average order value in the current view (**INR {high_region_value:,.0f}**).")
    st.subheader("Monthly sales")
    monthly=base.set_index("Order_Date").resample("MS").Sales.sum()
    st.line_chart(monthly)
with tabs[1]:
    st.subheader("Missing values, duplicates, and invalid entries")
    miss=pd.DataFrame({"Missing count":raw.isna().sum(),"Missing %":100*raw.isna().mean()}).query("`Missing count` > 0").sort_values("Missing %",ascending=False)
    st.dataframe(miss.style.format({"Missing %":"{:.2f}%"}),width="stretch")
    st.write(f"Exact duplicate rows detected: **{len(duplicates):,}**. Views exclude later copies by default; download preserves the source row and duplicate flag.")
    issues={"Age outside 0 to 120":int(cleaned.Flag_Invalid_Age.sum()),"Discount outside 0 to 0.5":int(cleaned.Flag_Invalid_Discount.sum()),"Rating outside 1 to 5":int(cleaned.Flag_Invalid_Rating.sum()),"Negative income":int(cleaned.Flag_Invalid_Income.sum()),"Quantity at or below zero":int(cleaned.Flag_Invalid_Quantity.sum()),"Unit price at or below zero":int(cleaned.Flag_Invalid_Unit_Price.sum()),"Negative sales":int(cleaned.Flag_Negative_Sales.sum()),"Order dates missing or invalid":int(cleaned.Flag_Date_Order.sum()),"Delivery dates missing or invalid":int(cleaned.Flag_Date_Delivery.sum()),"Non-ISO order date format":int(cleaned.Flag_Date_Format_Order.sum()),"Non-ISO delivery date format":int(cleaned.Flag_Date_Format_Delivery.sum()),"Delivery before order":int(cleaned.Flag_Delivery_Before_Order.sum())}
    st.dataframe(pd.DataFrame(issues.items(),columns=["Check","Rows flagged"]),hide_index=True,width="stretch")
    for col in ["Gender","Region"]:
        vals=raw[col].fillna("<missing>").astype(str).value_counts()
        st.write(f"**{col} source labels**"); st.dataframe(vals.rename("Count").to_frame(),width="stretch")
    for col in ["Sales","Profit","Age","Income"]:
        flags=iqr_flags(base[col]); st.write(f"**{col}: {int(flags.sum()):,} IQR outliers** of {base[col].notna().sum():,} valid observations. IQR flags point to values worth reviewing; they do not prove an error.")
with tabs[2]:
    st.subheader("Descriptive statistics")
    st.caption("Mean and standard deviation can be pulled by extreme values, so compare them with the median and quartiles.")
    stat=[]
    for col in NUMERIC:
        s=base[col].dropna(); modes=s.mode()
        stat.append({"Variable":col,"N":len(s),"Mean":s.mean(),"Median":s.median(),"Mode":modes.iloc[0] if len(modes) else np.nan,"Std dev":s.std(),"Variance":s.var(),"Q1":s.quantile(.25),"Q3":s.quantile(.75),"IQR":s.quantile(.75)-s.quantile(.25),"P5":s.quantile(.05),"P95":s.quantile(.95),"Skew":s.skew()})
    st.dataframe(pd.DataFrame(stat).set_index("Variable").style.format("{:.2f}"),width="stretch")
    num=st.selectbox("Numeric distribution",["Age","Income","Sales","Profit","Quantity","Unit_Price","Discount","Cost"])
    fig,ax=plt.subplots(figsize=(8,4)); sns.histplot(base[num].dropna(),kde=True,ax=ax,color="#276678"); ax.set_title(f"{num} distribution"); st.pyplot(fig); plt.close(fig)
    cat=st.selectbox("Category frequency",CATS)
    freq=base[cat].value_counts(dropna=False).rename_axis(cat).to_frame("Count"); freq["Proportion"]=freq.Count/freq.Count.sum(); st.dataframe(freq.style.format({"Proportion":"{:.1%}"}),width="stretch")
with tabs[3]:
    st.subheader("Business questions")
    questions={
      "Which region has the highest average order value?":base.groupby("Region").Sales.agg(Transactions="count",Average_Order_Value="mean",Total_Sales="sum").sort_values("Average_Order_Value",ascending=False),
      "Which product category has the lowest profit margin?":base.groupby("Product_Category").agg(Sales=("Sales","sum"),Profit=("Profit","sum")).assign(Profit_Margin=lambda x:x.Profit/x.Sales).sort_values("Profit_Margin"),
      "Which customer segment contributes the most sales?":base.groupby("Customer_Segment").Sales.agg(Transactions="count",Sales="sum",Average_Order_Value="mean").sort_values("Sales",ascending=False),
      "Which payment method has the largest average transaction?":base.groupby("Payment_Method").Sales.agg(Transactions="count",Average_Order_Value="mean").sort_values("Average_Order_Value",ascending=False),
      "What is the return rate by product category?":base.assign(Returned_Flag=base.Returned.str.casefold().eq("yes")).groupby("Product_Category").Returned_Flag.agg(Orders="count",Return_Rate="mean").sort_values("Return_Rate",ascending=False),
      "How does delivery speed vary by mode?":base.groupby("Delivery_Mode").Delivery_Days.agg(Orders="count",Average_Delivery_Days="mean",Median_Delivery_Days="median").sort_values("Average_Delivery_Days")}
    for q,table in questions.items():
        st.markdown("**"+q+"**"); st.dataframe(table,width="stretch")
        st.caption("Figures follow the filters above. Group sizes are shown so small groups are easy to spot.")
with tabs[4]:
    st.subheader("What changed in the analysis copy")
    st.markdown("""
- Exact duplicate source rows are flagged; analysis excludes later exact copies. Original records remain available through the duplicate toggle and source CSV.
- Gender and Region are normalized with explicit aliases (`M`/`F`, case variants, `Nort`, `Sth`). Unmapped labels are title-cased and should be reviewed.
- Dates are parsed from mixed formats. Non-ISO formats are counted, missing or invalid dates stay blank, and delivery-before-order rows are flagged and excluded from delivery duration. Ambiguous slash dates use month-first parsing; check these against source records before operational use.
- Missing age, income, discount, rating, and satisfaction values are not imputed: these are not needed to preserve the row, and median/mode imputation would invent customer or transaction facts. They remain missing and are counted.
- Ages outside 0 to 120, negative income, quantity or unit price at or below zero, negative sales, discounts outside 0 to 0.5, and ratings outside 1 to 5 are flagged and set missing in analysis fields. Other numeric extremes are retained and flagged by IQR for review, because a large order may be genuine.
- Engineered fields: `Profit_Margin = Profit / Sales`; `Delivery_Days = Delivery_Date - Order_Date`; `Revenue_per_Customer` is the customer's total Sales across the loaded dataset.
""")
    st.dataframe(base.head(100),width="stretch")
    full_cleaned = cleaned.loc[~cleaned.Flag_Duplicate].copy()
    st.download_button("Download full cleaned dataset",full_cleaned.to_csv(index=False).encode("utf-8-sig"),"retail_sales_cleaned.csv","text/csv")
    st.download_button("Download current filtered view",base.to_csv(index=False).encode("utf-8-sig"),"retail_sales_filtered_view.csv","text/csv")
    st.download_button("Download missing-value audit CSV",pd.DataFrame({"column":raw.columns,"missing_count":raw.isna().sum().values,"missing_percent":100*raw.isna().mean().values}).to_csv(index=False).encode("utf-8-sig"),"data_quality_audit.csv","text/csv")
    st.download_button("Download duplicate rows CSV",duplicates.to_csv(index=False).encode("utf-8-sig"),"duplicate_rows.csv","text/csv")

with st.expander("About the analysis"):
    st.write("This project describes the supplied transactions. An IQR flag marks an unusual value for review; it does not tell us whether the record is wrong.")

