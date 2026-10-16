"""
Data preprocessing and feature engineering for fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from config import RANDOM_SEED, TEST_SIZE

# يهدف هذا الملف إلى تجهيز بيانات المعاملات المالية قبل
# تدريب نماذج التعلم الآلي من خلال:
# 1- إنشاء خصائص جديدة (Feature Engineering).
# 2- ترميز المتغيرات الفئوية.
# 3- اختيار المتغيرات المستقلة والمتغير الهدف.
# 4- تقسيم البيانات إلى تدريب واختبار.
# 5- توحيد نطاق البيانات (Feature Scaling).
# 6- معالجة مشكلة عدم توازن الفئات.


# Feature Engineering
# إنشاء مجموعة من الخصائص الجديدة التي تساعد النموذج
# على اكتشاف أنماط الاحتيال بشكل أفضل
# الخصائص المضافة تشمل:
# - خصائص مرتبطة بقيمة المعاملة
# - فروقات الأرصدة قبل وبعد المعاملة
# - نسب المبلغ إلى الرصيد
# - مؤشرات ثنائية للحالات المشبوهة
# - مؤشرات المعاملات الكبيرة
# - أخطاء حساب الأرصدة
# - خصائص زمنية (اليوم والساعة)

# Categorical Encoding
# تحويل نوع المعاملة (Transaction Type)
# من بيانات نصية إلى بيانات رقمية يمكن للنموذج التعامل معها.
#
# تم استخدام:
# 1- Label Encoding
# 2- One-Hot Encoding


# prepare_features
# تجهيز مصفوفة الخصائص (X) والمتغير الهدف (y).
# يتم استبعاد الأعمدة التي لا تضيف قيمة تنبؤية مثل:
# - معرفات الحسابات.
# - المتغير الهدف.
# - الأعمدة النصية غير المستخدمة.

# Feature Scaling
# تطبيق StandardScaler لتوحيد نطاق جميع الخصائص.
# الهدف:
# - منع تأثير اختلاف المقاييس بين المتغيرات.
# - تحسين أداء بعض خوارزميات التعلم الآلي.
#
# يتم التدريب على بيانات Train فقط
# لتجنب تسرب البيانات (Data Leakage).

# Class Imbalance Handling
# معالجة مشكلة عدم توازن البيانات بين:
# - المعاملات الطبيعية.
# - المعاملات الاحتيالية.
#
# الطرق المدعومة:
# 1- Random Under Sampling
# 2- SMOTE Over Sampling
# 3- Combined Sampling
#
# يتم تطبيق المعالجة على بيانات التدريب فقط.

# Complete Preprocessing Pipeline
# تنفيذ جميع مراحل تجهيز البيانات بالترتيب:
#
# 1- Feature Engineering
# 2- Encoding
# 3- Feature Selection
# 4- Train/Test Split
# 5- Feature Scaling
# 6- Class Balancing
#
# المخرجات:
# - بيانات التدريب الجاهزة للنمذجة.
# - بيانات الاختبار.
# - أسماء الخصائص.
# - Scaler المستخدم.
# - Label Encoder المستخدم.

def engineer_features(df):
    """
    Create new features for fraud detection.

    Args:
        df: pandas DataFrame with raw data

    Returns:
        DataFrame with engineered features
    """
    print("\nEngineering features...")
    df = df.copy()


    # تحويل قيمة المعاملة إلى مقياس لوغاريتمي لتقليل تأثير
    # القيم الكبيرة جداً وتحسين توزيع البيانات.
    df['amount_log'] = np.log1p(df['amount'])

    # الفرق بين الرصيد بعد المعاملة والرصيد قبلها للحساب المرسل.
    # يساعد في كشف السلوك غير الطبيعي في عمليات السحب والتحويل.
    df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']

    # الفرق بين الرصيد بعد المعاملة والرصيد قبلها للحساب المستلم.
    # يستخدم لاكتشاف التناقضات في انتقال الأموال.
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # نسبة قيمة المعاملة إلى الرصيد الأصلي للمرسل.
    # تشير إلى مدى تأثير العملية على رصيد العميل.
    df['amount_to_orig_balance'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    # نسبة قيمة المعاملة إلى رصيد الحساب المستلم.
    # تساعد في تمييز التحويلات الكبيرة غير المعتادة.
    df['amount_to_dest_balance'] = df['amount'] / (df['oldbalanceDest'] + 1)

    # مؤشر ثنائي يساوي 1 إذا أصبح رصيد المرسل صفراً بعد العملية.
    # يعتبر نمطاً شائعاً في بعض عمليات الاحتيال.
    df['orig_balance_zeroed'] = (
        (df['newbalanceOrig'] == 0) &
        (df['oldbalanceOrg'] > 0)
    ).astype(int)

    # مؤشر ثنائي يساوي 1 إذا لم يتغير رصيد المستلم بعد التحويل.
    # قد يدل على وجود خطأ أو سلوك مشبوه.
    df['dest_balance_unchanged'] = (
        df['oldbalanceDest'] == df['newbalanceDest']
    ).astype(int)

    # تحديد المعاملات الكبيرة جداً (أكبر من 95% من جميع المعاملات).
    # يساعد في كشف العمليات المالية غير الاعتيادية.
    amount_95 = df['amount'].quantile(0.95)
    df['is_large_transaction'] = (
        df['amount'] > amount_95
    ).astype(int)

    # مؤشر يساوي 1 إذا تم تحويل كامل رصيد الحساب دفعة واحدة.
    # يعتبر سلوكاً قد يرتبط بمحاولات الاحتيال.
    df['full_balance_transfer'] = (
        (df['amount'] == df['oldbalanceOrg']) &
        (df['oldbalanceOrg'] > 0)
    ).astype(int)

    # حساب الخطأ بين الرصيد المتوقع والرصيد الفعلي للمرسل.
    # يساعد في اكتشاف السجلات غير المنطقية أو الاحتيالية.
    df['error_balance_orig'] = (
        df['oldbalanceOrg']
        - df['amount']
        - df['newbalanceOrig']
    )

    # حساب الخطأ بين الرصيد المتوقع والرصيد الفعلي للمستلم.
    # أي اختلاف كبير قد يشير إلى نشاط غير طبيعي.
    df['error_balance_dest'] = (
        df['oldbalanceDest']
        + df['amount']
        - df['newbalanceDest']
    )


# ملاحظة ::ال 
# step 
# بالداتا هو عدد الساعات من بداية العملية اتناء تحميع الداتا 

    # استخراج ساعة تنفيذ المعاملة من متغير الزمن step.
    # يساعد في معرفة ما إذا كانت العمليات تحدث في أوقات غير معتادة.
    df['hour'] = df['step'] % 24

    # استخراج رقم اليوم من متغير الزمن.
    # يسمح بدراسة الأنماط اليومية للمعاملات.
    df['day'] = df['step'] // 24

    # تقسيم قيم المعاملات إلى 10 فئات (Deciles)
    # لتمثيل مستوى المبلغ بشكل نسبي بدلاً من القيمة الخام.
    df['amount_decile'] = pd.qcut(
        df['amount'],
        q=10,
        labels=False,
        duplicates='drop'
    )
    # تقسيم جميع المعاملات إلى 10 مجموعات متساوية تقريباً حسب قيمة المبلغ
    # بدال من 
# amount
# 100
# 500
# 1000
# 50000
# بيصير
#  amount  amount_decile 
#  100     0             
#  300     1             
#  700     2             
#  1200    3             
#  ...     ...           
#  50000   9             

# 0 → من أصغر المعاملات.
# 5 → معاملة متوسطة الحجم.
# 9 → من أكبر 10% من المعاملات.

# هذا يساعد النموذج على اكتشاف ما إذا كانت المعاملة تقع ضمن فئة المبالغ المرتفعة أو المنخفضة،
#  وهو أمر مفيد لأن عمليات الاحتيال غالباً ما ترتبط بأحجام معاملات معينة.
    print(f"Created {len(df.columns) - 11} new features")

    return df


def encode_categorical(df):
    """
    Encode categorical variables.

    Args:
        df: pandas DataFrame

    Returns:
        DataFrame with encoded categories, label encoder
    """
    df = df.copy()
    le = LabelEncoder()

    # Encode transaction type
    df['type_encoded'] = le.fit_transform(df['type'])

    # Create one-hot encoding for transaction type
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)

    return df, le


def prepare_features(df, drop_columns=None):
    """
    تجهيز مصفوفة الخصائص (Features) والمتغير الهدف (Target)
    لاستخدامها في تدريب نماذج التعلم الآلي.
    """

    # الأعمدة التي لن يتم استخدامها كمدخلات للنموذج
    # isFraud : المتغير الهدف الذي نريد التنبؤ به
    # isFlaggedFraud : متغير قد يسبب تسرب معلومات للنموذج
    # nameOrig و nameDest : معرفات حسابات لا تحمل قيمة تنبؤية مباشرة
    # type : تم ترميزه مسبقاً لذلك لا حاجة لاستخدام النص الأصلي
    exclude_cols = [
        'isFraud',
        'isFlaggedFraud',
        'nameOrig',
        'nameDest',
        'type'
    ]

    # السماح باستبعاد أعمدة إضافية عند الحاجة
    if drop_columns:
        exclude_cols.extend(drop_columns)

    # اختيار جميع الأعمدة التي سيتم استخدامها كخصائص للنموذج
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
    ]

    # إنشاء مصفوفة الخصائص (Features)
    X = df[feature_cols]

    # إنشاء المتغير الهدف (Target)
    # 0 = معاملة طبيعية
    # 1 = معاملة احتيالية
    y = df['isFraud']

    # عرض حجم البيانات وعدد الخصائص
    print(f"\nFeature matrix shape: {X.shape}")

    # عرض توزيع الفئات لمعرفة درجة عدم التوازن
    print(f"Target distribution:\n{y.value_counts()}")

    # إرجاع الخصائص والهدف وأسماء الخصائص
    return X, y, feature_cols

def scale_features(X_train, X_test, feature_names):
    """
    Scale numerical features using StandardScaler.

    Args:
        X_train, X_test: training and test feature matrices
        feature_names: list of feature names

    Returns:
        Scaled X_train, X_test, scaler object
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


def handle_imbalance(X_train, y_train, sampling_strategy=0.1):
    """
    معالجة عدم توازن البيانات باستخدام
    Random Under Sampling فقط.

    Parameters:
        X_train : بيانات التدريب
        y_train : الفئات المستهدفة
        sampling_strategy : نسبة الاحتفاظ من الفئة الكبرى

    Returns:
        X_resampled, y_resampled
    """

    print("\nHandling class imbalance using Random Under Sampling...")
    print(f"Before resampling: {y_train.value_counts().to_dict()}")

    sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_SEED
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    print(
        f"After resampling: "
        f"{pd.Series(y_resampled).value_counts().to_dict()}"
    )

    return X_resampled, y_resampled

def preprocess_pipeline(df,test_size=TEST_SIZE):
    """
    Complete preprocessing pipeline.

    Args:
        df: raw DataFrame
        test_size: fraction for test set
        handle_imbalance_method: method to handle class imbalance

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)

    # Feature engineering
    df = engineer_features(df)

    # Encode categorical variables
    df, label_encoder = encode_categorical(df)

    # Prepare features
    X, y, feature_names = prepare_features(df)

    # Split data
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Scale features
    print("\nScaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_names)

    # معالجة عدم توازن البيانات باستخدام Random Under Sampling
# يتم تطبيقها على بيانات التدريب فقط لتجنب التأثير على بيانات الاختبار
    X_train_balanced, y_train_balanced = handle_imbalance(
    X_train_scaled,
    y_train
)
    print("\nPreprocessing complete!")

    return {
        'X_train': X_train_balanced,
        'X_test': X_test_scaled,
        'y_train': y_train_balanced,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'label_encoder': label_encoder
    }


if __name__ == "__main__":
    from data_loader import load_data

    # Test with sample
    df = load_data(sample_size=100000)
    result = preprocess_pipeline(df)
    print(f"\nFinal training set shape: {result['X_train'].shape}")
    print(f"Final test set shape: {result['X_test'].shape}")
