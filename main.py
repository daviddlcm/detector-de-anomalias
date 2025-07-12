import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
from prophet  import Prophet
import warnings
warnings.filterwarnings('ignore')

class SolarAnomalyDetector:
    def __init__(self):
        self.df_combined = None
        self.inversor_data = None
        self.anomalies_euclidean = None
        self.anomalies_mahalanobis = None
        self.anomalies_isolation = None
        self.anomalies_prophet = None  # <-- AÑADIDO
        
    def load_and_merge_datasets(self, electrical_path, environment_path, irradiance_path):
        """
        Carga y combina los tres datasets usando measured_on como clave
        """
        print("=== CARGA Y DESCRIPCIÓN DE DATOS ===")
        
        # Cargar los datasets
        electrical_df = pd.read_csv(electrical_path)
        environment_df = pd.read_csv(environment_path)
        irradiance_df = pd.read_csv(irradiance_path)
        
        print(f"Dataset eléctrico: {electrical_df.shape}")
        print(f"Dataset ambiental: {environment_df.shape}")
        print(f"Dataset irradiancia: {irradiance_df.shape}")
        
        # Convertir measured_on a datetime
        electrical_df['measured_on'] = pd.to_datetime(electrical_df['measured_on'])
        environment_df['measured_on'] = pd.to_datetime(environment_df['measured_on'])
        irradiance_df['measured_on'] = pd.to_datetime(irradiance_df['measured_on'])
        
        # Combinar los datasets
        print("Combinando datasets...")
        df_combined = electrical_df.merge(environment_df, on='measured_on', how='inner')
        df_combined = df_combined.merge(irradiance_df, on='measured_on', how='inner')
        
        print(f"Dataset combinado: {df_combined.shape}")
        print(f"Columnas disponibles: {list(df_combined.columns)}")
        
        # Verificar valores faltantes y duplicados
        print("\n=== IDENTIFICACIÓN DE VALORES FALTANTES Y DUPLICADOS ===")
        missing_values = df_combined.isnull().sum()
        duplicated_rows = df_combined.duplicated().sum()
        
        print(f"Valores faltantes por columna:")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"  - {col}: {missing} ({missing/len(df_combined)*100:.2f}%)")
        
        print(f"Filas duplicadas: {duplicated_rows}")
        
        self.df_combined = df_combined
        return df_combined
    
    def exploratory_data_analysis(self, df):
        """
        Análisis exploratorio de datos
        """
        print("\n=== ANÁLISIS EXPLORATORIO DE DATOS ===")
        
        # Estructura de los datos
        print(f"Forma del dataset: {df.shape}")
        print(f"Tipos de datos:")
        print(df.dtypes.value_counts())
        
        # Estadísticas descriptivas para variables numéricas
        print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Variables numéricas encontradas: {len(numeric_cols)}")
        
        # Mostrar estadísticas para las primeras 10 columnas numéricas
        print("\nEstadísticas descriptivas (primeras 10 variables numéricas):")
        print(df[numeric_cols[:10]].describe())
        
        # Crear histogramas para variables continuas principales
        self.create_histograms(df, numeric_cols[:6])  # Primeras 6 para no saturar
        
        # Análisis de correlaciones
        self.analyze_correlations(df, numeric_cols[:15])  # Primeras 15 para matriz legible
        
    def create_histograms(self, df, columns):
        """
        Crear histogramas para variables continuas
        """
        print("\n=== CREANDO HISTOGRAMAS ===")
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, col in enumerate(columns):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribución de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_correlations(self, df, columns):
        """
        Analizar correlaciones entre variables
        """
        print("\n=== ANÁLISIS DE CORRELACIONES ===")
        
        # Matriz de correlación
        correlation_matrix = df[columns].corr()
        
        # Visualizar matriz de correlación
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.show()
        
        # Identificar correlaciones altas
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        corr_value
                    ))
        
        print(f"Pares de variables con alta correlación (|r| > 0.7): {len(high_corr_pairs)}")
        for var1, var2, corr in high_corr_pairs[:10]:  # Mostrar solo los primeros 10
            print(f"  - {var1} vs {var2}: r = {corr:.3f}")
    
    def clean_and_preprocess_data(self, df):
        """
        Limpia y preprocesa los datos siguiendo la estrategia definida.
        VERSIÓN CORREGIDA.
        """
        print("\n=== LIMPIEZA Y PREPROCESAMIENTO ===")
        
        # 1. Renombrar columnas primero en el DF combinado
        # Esto facilita la selección después
        columnas_renombrar = {}
        for columna in df.columns:
            if '_inv_' in columna:
                nueva_columna = 'inv_' + columna.split('_inv_')[0]
            elif '_o_' in columna:
                nueva_columna = columna.split('_o_')[0]
            else:
                nueva_columna = columna
            # Corregir error común de tipeo
            if 'irridiance' in nueva_columna:
                nueva_columna = nueva_columna.replace('irridiance', 'irradiance')
            columnas_renombrar[columna] = nueva_columna
        
        df_renamed = df.rename(columns=columnas_renombrar)
        
        # 2. Filtrar por el primer inversor disponible
        # Buscamos columnas que ahora se llaman 'inv_inv_XX_dc_current'
        inversores_disponibles = sorted(list(set([c.split('_')[2] for c in df_renamed.columns if c.startswith('inv_inv_')])))
        
        if not inversores_disponibles:
            print("No se encontraron inversores disponibles después de renombrar.")
            return None
            
        inversor_id = inversores_disponibles[0]
        print(f"Procesando datos para el inversor: {inversor_id}")
        
        # 3. Seleccionar todas las columnas relevantes en una sola pasada ### <<< CAMBIO CLAVE
        columnas_inversor = [c for c in df_renamed.columns if f'inv_{inversor_id}' in c]
        columnas_ambientales = ['ambient_temperature', 'wind_speed', 'wind_direction', 'poa_irradiance']
        columnas_finales = ['measured_on'] + columnas_inversor + [c for c in columnas_ambientales if c in df_renamed.columns]
        
        # Crear el DataFrame del inversor con todas las columnas necesarias
        inversor_data = df_renamed[columnas_finales].copy()
        
        # 4. Limpiar nombres de columnas específicos del inversor ### <<< CAMBIO CLAVE
        nombres_finales = {}
        prefijo = f'inv_inv_{inversor_id}_'
        for col in inversor_data.columns:
            if col.startswith(prefijo):
                nombres_finales[col] = col.replace(prefijo, '')
        inversor_data = inversor_data.rename(columns=nombres_finales)

        # 5. Procesamiento de la serie temporal (ahora con todas las columnas juntas) ### <<< CAMBIO CLAVE
        inversor_data['measured_on'] = pd.to_datetime(inversor_data['measured_on'])
        inversor_data = inversor_data.set_index('measured_on')
        
        # Filtrar datos nocturnos (6 AM a 6 PM)
        print("Filtrando datos nocturnos...")
        inversor_data = inversor_data.between_time('06:00', '18:00')
        
        # Resample a promedios horarios
        print("Aplicando promedios horarios...")
        inversor_data = inversor_data.resample('1H').mean()
        
        # Interpolación lineal para valores faltantes
        print("Interpolando valores faltantes...")
        inversor_data = inversor_data.interpolate(method='linear')
        
        # Eliminar filas con valores faltantes
        inversor_data = inversor_data.dropna()
        
        print(f"Datos procesados: {inversor_data.shape}")
        print(f"Columnas finales disponibles: {list(inversor_data.columns)}")
        print(f"Período: {inversor_data.index.min()} a {inversor_data.index.max()}")
        
        self.inversor_data = inversor_data
        return inversor_data
    
    def obtener_inversores_disponibles(self, df, var_type, var_name):
        """Obtiene la lista de inversores disponibles"""
        inversores = set()
        for columna in df.columns:
            if columna.startswith('inv_') and var_type in columna and var_name in columna:
                try:
                    numero = columna.split('_')[1]
                    inversores.add(int(numero))
                except (IndexError, ValueError):
                    continue
        return sorted(list(inversores))
    
    def filtrar_por_inversor(self, df, numero_inversor):
        numero_formateado = f"{numero_inversor:02d}"
        columnas_inversor = [col for col in df.columns if col.startswith(f"inv_{numero_formateado}_")]
        columnas_finales = ['measured_on'] + columnas_inversor
        return df[columnas_finales]
    
    def limpiar_nombres_columnas(self, df):
        columnas_limpias = {}
        for columna in df.columns:
            if columna == 'measured_on':
                columnas_limpias[columna] = columna
            elif columna.startswith('inv_'):
                partes = columna.split('_')
                if len(partes) >= 3:
                    nuevo_nombre = '_'.join(partes[2:])
                    columnas_limpias[columna] = nuevo_nombre
                else:
                    columnas_limpias[columna] = columna
            else:
                columnas_limpias[columna] = columna
        return df.rename(columns=columnas_limpias)
    
    def temporal_analysis(self, df):
        """
        Análisis temporal de los datos
        """
        print("\n=== ANÁLISIS TEMPORAL ===")
        
        # Análisis por hora del día
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Patrones por hora
        hourly_patterns = df.groupby('hour').mean()
        
        # Seleccionar algunas variables para visualización
        power_cols = [col for col in df.columns if 'power' in col.lower()][:3]
        
        if power_cols:
            plt.figure(figsize=(15, 8))
            
            plt.subplot(2, 2, 1)
            for col in power_cols:
                plt.plot(hourly_patterns.index, hourly_patterns[col], label=col, marker='o')
            plt.title('Patrones por Hora del Día')
            plt.xlabel('Hora')
            plt.ylabel('Potencia Promedio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Patrones por día de la semana
            plt.subplot(2, 2, 2)
            weekly_patterns = df.groupby('day_of_week').mean()
            for col in power_cols:
                plt.plot(weekly_patterns.index, weekly_patterns[col], label=col, marker='o')
            plt.title('Patrones por Día de la Semana')
            plt.xlabel('Día de la Semana (0=Lunes)')
            plt.ylabel('Potencia Promedio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Patrones por mes
            plt.subplot(2, 2, 3)
            monthly_patterns = df.groupby('month').mean()
            for col in power_cols:
                plt.plot(monthly_patterns.index, monthly_patterns[col], label=col, marker='o')
            plt.title('Patrones por Mes')
            plt.xlabel('Mes')
            plt.ylabel('Potencia Promedio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Serie temporal general
            plt.subplot(2, 2, 4)
            sample_data = df[power_cols[0]].resample('D').mean()
            plt.plot(sample_data.index, sample_data.values)
            plt.title(f'Serie Temporal Diaria - {power_cols[0]}')
            plt.xlabel('Fecha')
            plt.ylabel('Potencia Promedio Diaria')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Limpiar columnas temporales
        df.drop(['hour', 'day_of_week', 'month'], axis=1, inplace=True)
    
    def feature_engineering(self, df):
        """
        Ingeniería de características
        """
        print("\n=== INGENIERÍA DE CARACTERÍSTICAS ===")
        
        # Características básicas ya están en el dataset
        # Agregar algunas características derivadas si es necesario
        
        # Eficiencia de conversión (si tenemos tanto AC como DC power)
        power_ac_cols = [col for col in df.columns if 'power' in col.lower() and 'ac' in col.lower()]
        power_dc_cols = [col for col in df.columns if 'power' in col.lower() and 'dc' in col.lower()]
        
        if power_ac_cols and power_dc_cols:
            # Tomar la primera columna de cada tipo
            ac_col = power_ac_cols[0]
            dc_col = power_dc_cols[0]
            
            # Calcular eficiencia (AC/DC)
            df['efficiency'] = df[ac_col] / (df[dc_col] + 1e-8)  # Evitar división por cero
            print(f"Característica creada: efficiency = {ac_col} / {dc_col}")
        
        # Ratios de corriente y voltaje
        current_cols = [col for col in df.columns if 'current' in col.lower()]
        voltage_cols = [col for col in df.columns if 'voltage' in col.lower()]
        
        if len(current_cols) >= 2:
            df['current_ratio'] = df[current_cols[0]] / (df[current_cols[1]] + 1e-8)
            print(f"Característica creada: current_ratio = {current_cols[0]} / {current_cols[1]}")
        
        if len(voltage_cols) >= 2:
            df['voltage_ratio'] = df[voltage_cols[0]] / (df[voltage_cols[1]] + 1e-8)
            print(f"Característica creada: voltage_ratio = {voltage_cols[0]} / {voltage_cols[1]}")
        
        return df
    
    def select_variables(self, df):
        """
        Selección de variables para detección de anomalías
        """
        print("\n=== SELECCIÓN DE VARIABLES ===")
        
        # Variables eléctricas importantes
        electrical_vars = []
        for col in df.columns:
            if any(term in col.lower() for term in ['power', 'current', 'voltage', 'efficiency']):
                electrical_vars.append(col)
        
        # Variables ambientales
        environmental_vars = []
        for col in df.columns:
            if any(term in col.lower() for term in ['temperature', 'wind_speed', 'irradiance', 'wind_direction']):
                environmental_vars.append(col)
        
        # Variables derivadas
        derived_vars = []
        for col in df.columns:
            if any(term in col.lower() for term in ['ratio', 'efficiency']):
                derived_vars.append(col)
        
        # Combinar variables seleccionadas
        selected_vars = electrical_vars + environmental_vars + derived_vars
        
        # Si no hay variables específicas, usar todas las numéricas
        if not selected_vars:
            selected_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Variables eléctricas seleccionadas ({len(electrical_vars)}): {electrical_vars}")
        print(f"Variables ambientales seleccionadas ({len(environmental_vars)}): {environmental_vars}")
        print(f"Variables derivadas seleccionadas ({len(derived_vars)}): {derived_vars}")
        
        print(f"\nJustificación de la selección:")
        print("- Variables eléctricas: Indican el rendimiento directo del sistema solar")
        print("- Variables ambientales: Afectan directamente la generación de energía")
        print("- Variables derivadas: Proporcionan información sobre eficiencia y ratios")
        
        return df[selected_vars]
    
    def euclidean_distance_detector(self, df, percentile=95):
        """
        Detector de anomalías basado en distancia euclidiana
        """
        print(f"\n=== DETECTOR EUCLIDIANO (percentil {percentile}%) ===")
        
        # Verificar que tenemos datos
        if df.empty:
            print("Error: No hay datos para procesar")
            return None
        
        # Normalizar datos
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        
        # Calcular distancias euclidianas
        datos = df_scaled.values
        centro = np.mean(datos, axis=0)
        distancias = np.sqrt(np.sum((datos - centro)**2, axis=1))
        
        # Crear serie con las distancias
        distancias_series = pd.Series(distancias, index=df.index)
        
        # Determinar umbral y anomalías
        umbral = np.percentile(distancias, percentile)
        anomalias = distancias_series[distancias_series > umbral]
        
        print(f"Umbral calculado: {umbral:.4f}")
        print(f"Anomalías detectadas: {len(anomalias)}")
        print(f"Porcentaje de anomalías: {len(anomalias)/len(df)*100:.2f}%")
        
        # Visualizar
        plt.figure(figsize=(14, 6))
        plt.plot(distancias_series.index, distancias_series.values, label='Distancia euclidiana', alpha=0.7)
        plt.axhline(umbral, color='r', linestyle='--', label=f'Umbral (percentil {percentile}%)')
        plt.scatter(anomalias.index, anomalias.values, color='red', label='Anomalías', s=30, alpha=0.8)
        plt.title(f'Detector de Anomalías - Distancia Euclidiana')
        plt.xlabel('Fecha')
        plt.ylabel('Distancia euclidiana')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        self.anomalies_euclidean = anomalias
        return anomalias
    
    def mahalanobis_distance_detector(self, df, percentile=95):
        """
        Detector de anomalías basado en distancia de Mahalanobis
        """
        print(f"\n=== DETECTOR MAHALANOBIS (percentil {percentile}%) ===")
        
        # Verificar que tenemos datos
        if df.empty:
            print("Error: No hay datos para procesar")
            return None
        
        # Normalizar datos
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        
        # Calcular matriz de covarianza
        datos = df_scaled.values
        cov_matrix = np.cov(datos, rowvar=False)
        
        # Verificar que la matriz no es singular
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print("Matriz de covarianza singular, usando pseudo-inversa")
            inv_cov = np.linalg.pinv(cov_matrix)
        
        # Calcular centro de los datos
        centro = np.mean(datos, axis=0)
        
        # Calcular distancias de Mahalanobis
        distancias = []
        for i in range(len(datos)):
            diff = datos[i] - centro
            distancia = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
            distancias.append(distancia)
        
        # Crear serie con las distancias
        distancias_series = pd.Series(distancias, index=df.index)
        
        # Determinar umbral y anomalías
        umbral = np.percentile(distancias, percentile)
        anomalias = distancias_series[distancias_series > umbral]
        
        print(f"Umbral calculado: {umbral:.4f}")
        print(f"Anomalías detectadas: {len(anomalias)}")
        print(f"Porcentaje de anomalías: {len(anomalias)/len(df)*100:.2f}%")
        
        # Visualizar
        plt.figure(figsize=(14, 6))
        plt.plot(distancias_series.index, distancias_series.values, label='Distancia Mahalanobis', alpha=0.7)
        plt.axhline(umbral, color='r', linestyle='--', label=f'Umbral (percentil {percentile}%)')
        plt.scatter(anomalias.index, anomalias.values, color='red', label='Anomalías', s=30, alpha=0.8)
        plt.title(f'Detector de Anomalías - Distancia de Mahalanobis')
        plt.xlabel('Fecha')
        plt.ylabel('Distancia de Mahalanobis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        self.anomalies_mahalanobis = anomalias
        return anomalias
    
    def isolation_forest_detector(self, df, contamination=0.05):
        """
        Detector de anomalías basado en Isolation Forest
        """
        print(f"\n=== DETECTOR ISOLATION FOREST (contaminación {contamination*100}%) ===")
        
        # Verificar que tenemos datos
        if df.empty:
            print("Error: No hay datos para procesar")
            return None
        
        # Normalizar datos
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        
        # Aplicar Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100
        )
        anomaly_labels = iso_forest.fit_predict(df_scaled)
        
        # Obtener scores de anomalía
        anomaly_scores = iso_forest.decision_function(df_scaled)
        
        # Crear series
        anomaly_scores_series = pd.Series(anomaly_scores, index=df.index)
        
        # Identificar anomalías
        anomalias_mask = anomaly_labels == -1
        anomalias = anomaly_scores_series[anomalias_mask]
        
        print(f"Anomalías detectadas: {len(anomalias)}")
        print(f"Porcentaje de anomalías: {len(anomalias)/len(df)*100:.2f}%")
        
        if len(anomalias) > 0:
            umbral_efectivo = anomalias.max()
            print(f"Umbral efectivo: {umbral_efectivo:.4f}")
        else:
            umbral_efectivo = anomaly_scores_series.min()
            print(f"No se detectaron anomalías. Score mínimo: {umbral_efectivo:.4f}")
        
        # Visualizar
        plt.figure(figsize=(14, 6))
        plt.plot(anomaly_scores_series.index, anomaly_scores_series.values, 
                label='Anomaly Score', alpha=0.7, color='blue')
        plt.axhline(umbral_efectivo, color='r', linestyle='--', 
                   label=f'Umbral efectivo (≤{umbral_efectivo:.3f})')
        plt.scatter(anomalias.index, anomalias.values, color='red', 
                   label='Anomalías', s=30, alpha=0.8)
        plt.title(f'Detector de Anomalías - Isolation Forest')
        plt.xlabel('Fecha')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        self.anomalies_isolation = anomalias
        return anomalias
    
    def prophet_anomaly_detector(self, df, interval_width=0.99):
        """
        Detector de anomalías basado en el modelo de series temporales Prophet.
        """
        print(f"\n=== DETECTOR DE ANOMALÍAS CON PROPHET (intervalo de confianza {interval_width*100}%) ===")

        # Seleccionar una variable objetivo para el pronóstico (priorizar potencia)
        target_col = None
        power_cols = [col for col in df.columns if 'power' in col.lower()]
        if power_cols:
            target_col = power_cols[0]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]

        if not target_col:
            print("Error: No se encontró una variable objetivo adecuada para Prophet.")
            return None

        print(f"Variable objetivo para Prophet: {target_col}")

        # Preparar el DataFrame para Prophet (requiere columnas 'ds' y 'y')
        df_prophet = df[[target_col]].reset_index()
        df_prophet = df_prophet.rename(columns={'measured_on': 'ds', target_col: 'y'})

        # Inicializar y ajustar el modelo Prophet
        model = Prophet(interval_width=interval_width, daily_seasonality=True)
        model.fit(df_prophet)

        # Realizar la predicción
        forecast = model.predict(df_prophet[['ds']])

        # Combinar el pronóstico con los datos originales
        results = pd.concat([df_prophet.set_index('ds'), forecast.set_index('ds')], axis=1)

        # Identificar anomalías (valor real 'y' fuera del intervalo [yhat_lower, yhat_upper])
        anomalies_mask = (results['y'] < results['yhat_lower']) | (results['y'] > results['yhat_upper'])
        anomalies = results[anomalies_mask]

        self.anomalies_prophet = df[target_col][anomalies_mask]

        print(f"Anomalías detectadas: {len(anomalies)}")
        print(f"Porcentaje de anomalías: {len(anomalies) / len(df) * 100:.2f}%")

        # Visualizar los resultados
        fig = model.plot(forecast, figsize=(14, 6))
        plt.scatter(anomalies.index, anomalies['y'], color='red', s=30, label='Anomalías')
        plt.title(f'Detector de Anomalías con Prophet - {target_col}')
        plt.xlabel('Fecha')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Visualizar componentes del modelo
        model.plot_components(forecast)
        plt.show()

        return self.anomalies_prophet

    def compare_anomaly_methods(self):
        """
        Comparar resultados entre métodos
        """
        print("\n=== COMPARACIÓN DE MÉTODOS ===")
        
        # Número de anomalías por método
        n_euclidean = len(self.anomalies_euclidean) if self.anomalies_euclidean is not None else 0
        n_mahalanobis = len(self.anomalies_mahalanobis) if self.anomalies_mahalanobis is not None else 0
        n_isolation = len(self.anomalies_isolation) if self.anomalies_isolation is not None else 0
        n_prophet = len(self.anomalies_prophet) if self.anomalies_prophet is not None else 0
        
        print(f"Número de anomalías detectadas:")
        print(f"  - Distancia Euclidiana: {n_euclidean}")
        print(f"  - Distancia Mahalanobis: {n_mahalanobis}")
        print(f"  - Isolation Forest: {n_isolation}")
        print(f"  - Prophet: {n_prophet}")
        
        # Overlap entre métodos
        if all([self.anomalies_euclidean is not None, 
                self.anomalies_mahalanobis is not None, 
                self.anomalies_isolation is not None,
                self.anomalies_prophet is not None]):
            
            euclidean_dates = set(self.anomalies_euclidean.index)
            mahalanobis_dates = set(self.anomalies_mahalanobis.index)
            isolation_dates = set(self.anomalies_isolation.index)
            prophet_dates = set(self.anomalies_prophet.index)
            
            # Intersecciones
            all_methods = euclidean_dates.intersection(mahalanobis_dates).intersection(isolation_dates).intersection(prophet_dates)
            
            print(f"\nOverlap entre métodos:")
            print(f"  - Euclidiana ∩ Mahalanobis: {len(euclidean_dates.intersection(mahalanobis_dates))}")
            print(f"  - Euclidiana ∩ Isolation Forest: {len(euclidean_dates.intersection(isolation_dates))}")
            print(f"  - Euclidiana ∩ Prophet: {len(euclidean_dates.intersection(prophet_dates))}")
            print(f"  - Mahalanobis ∩ Isolation Forest: {len(mahalanobis_dates.intersection(isolation_dates))}")
            print(f"  - Mahalanobis ∩ Prophet: {len(mahalanobis_dates.intersection(prophet_dates))}")
            print(f"  - Isolation Forest ∩ Prophet: {len(isolation_dates.intersection(prophet_dates))}")
            print(f"  - Todos los métodos: {len(all_methods)}")
        
        # Análisis de sensibilidad
        print(f"\nINTERPRETACIÓN DE SENSIBILIDAD:")
        print(f"- Distancia Euclidiana: MENOS sensible a correlaciones, trata cada variable independientemente.")
        print(f"- Distancia Mahalanobis: MÁS sensible a correlaciones, considera la covarianza.")
        print(f"- Isolation Forest: Sensible a la estructura de los datos, puede capturar interacciones no lineales.")
        print(f"- Prophet: Sensible a desviaciones de patrones temporales (tendencias y estacionalidades).")
    
    def generate_anomaly_dates_list(self):
        """
        Generar lista de fechas de anomalías detectadas
        """
        print("\n=== FECHAS DE ANOMALÍAS DETECTADAS ===")
        
        results = {}
        
        methods = [
            ("Distancia Euclidiana", self.anomalies_euclidean),
            ("Distancia de Mahalanobis", self.anomalies_mahalanobis),
            ("Isolation Forest", self.anomalies_isolation),
            ("Prophet", self.anomalies_prophet)
        ]

        for name, anomalies in methods:
            if anomalies is not None:
                dates = anomalies.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                results[name] = dates
                print(f"\n{name} ({len(dates)} anomalías):")
                for i, date in enumerate(dates[:5]):
                    print(f"  - {date}")
                if len(dates) > 5:
                    print(f"  ... y {len(dates) - 5} más")
        
        return results
    
    def analyze_anomaly_context(self):
        """
        Analizar el contexto y coherencia de las anomalías detectadas
        """
        print("\n=== ANÁLISIS DE CONTEXTO DE ANOMALÍAS ===")
        
        if self.inversor_data is None:
            print("No hay datos disponibles para análisis de contexto")
            return
        
        methods = [
            ("Euclidiana", self.anomalies_euclidean),
            ("Mahalanobis", self.anomalies_mahalanobis),
            ("Isolation Forest", self.anomalies_isolation),
            ("Prophet", self.anomalies_prophet)
        ]

        for name, anomalies in methods:
            if anomalies is not None and len(anomalies) > 0:
                print(f"\n--- Análisis de {name} ---")
                
                # Patrones por hora
                anomaly_hours = anomalies.index.hour
                hour_counts = pd.Series(anomaly_hours).value_counts().sort_index()
                print(f"Distribución por hora del día:")
                for hour, count in hour_counts.head(5).items():
                    print(f"  {hour:02d}:00 - {count} anomalías")
                
                # Patrones por día de la semana
                anomaly_days = anomalies.index.dayofweek
                day_counts = pd.Series(anomaly_days).value_counts().sort_index()
                days = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
                print(f"Distribución por día de la semana:")
                for day_idx, count in day_counts.items():
                    print(f"  {days[day_idx]} - {count} anomalías")
    
    def create_summary_visualization(self):
        """
        Crear visualización resumen de todas las anomalías
        """
        print("\n=== CREANDO VISUALIZACIÓN RESUMEN ===")
        
        if self.inversor_data is None:
            print("No hay datos disponibles para visualización")
            return
        
        power_cols = [col for col in self.inversor_data.columns if 'power' in col.lower()]
        target_col = power_cols[0] if power_cols else self.inversor_data.select_dtypes(include=[np.number]).columns[0]
        
        if target_col:
            plt.figure(figsize=(16, 8))
            
            sample_data = self.inversor_data[target_col].resample('D').mean()
            plt.plot(sample_data.index, sample_data.values, 
                    label=f'{target_col} (Promedio diario)', alpha=0.7)
            
            colors = ['red', 'orange', 'purple', 'green']
            methods = [
                ('Euclidiana', self.anomalies_euclidean),
                ('Mahalanobis', self.anomalies_mahalanobis),
                ('Isolation Forest', self.anomalies_isolation),
                ('Prophet', self.anomalies_prophet)
            ]
            
            for i, (name, anomalies) in enumerate(methods):
                if anomalies is not None and len(anomalies) > 0:
                    anomaly_days = anomalies.index.floor('D').unique()
                    plt.scatter(anomaly_days, sample_data.reindex(anomaly_days), 
                                color=colors[i], label=name, s=50, alpha=0.8)
            
            plt.title('Resumen de Anomalías Detectadas por Todos los Métodos')
            plt.xlabel('Fecha')
            plt.ylabel(f'{target_col}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    
    def analyze_bivariate_relationships(self, df):
        """
        Analiza y visualiza las relaciones entre pares de variables clave.
        """
        print("\n=== ANÁLISIS DE RELACIONES BIVARIADAS ===")
        
        # Verificar que las columnas necesarias existan
        required_cols = ['poa_irradiance', 'ac_power', 'ambient_temperature', 'efficiency']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if not available_cols:
            print("No se encontraron las columnas necesarias para el análisis bivariado.")
            return

        # Crear una figura para poner todas las gráficas
        num_plots = len(available_cols) -1 # No graficamos eficiencia vs si misma
        if num_plots <= 0 : return
        
        plt.figure(figsize=(18, 5))

        # 1. Gráfico de Potencia vs. Irradiancia
        if 'poa_irradiance' in available_cols and 'ac_power' in available_cols:
            plt.subplot(1, 3, 1)
            sns.scatterplot(data=df, x='poa_irradiance', y='ac_power', alpha=0.4, s=10)
            plt.title('Potencia AC vs. Irradiancia POA')
            plt.xlabel('Irradiancia (W/m^2)')
            plt.ylabel('Potencia AC (kW)')
            plt.grid(True)

        # 2. Gráfico de Potencia vs. Temperatura
        if 'ambient_temperature' in available_cols and 'ac_power' in available_cols:
            plt.subplot(1, 3, 2)
            sns.scatterplot(data=df, x='ambient_temperature', y='ac_power', alpha=0.4, s=10)
            plt.title('Potencia AC vs. Temperatura Ambiente')
            plt.xlabel('Temperatura Ambiente (°C)')
            plt.ylabel('Potencia AC (kW)')
            plt.grid(True)
        
        # 3. Gráfico de Eficiencia vs. Irradiancia (si existe la columna eficiencia)
        if 'efficiency' in available_cols and 'poa_irradiance' in available_cols:
            plt.subplot(1, 3, 3)
            # Filtrar valores de eficiencia extremos para una mejor visualización
            df_filtered = df[df['efficiency'].between(0.5, 1.0)] 
            sns.scatterplot(data=df_filtered, x='poa_irradiance', y='efficiency', alpha=0.4, s=10)
            plt.title('Eficiencia vs. Irradiancia')
            plt.xlabel('Irradiancia (W/m^2)')
            plt.ylabel('Eficiencia de Conversión')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()
    
    def export_anomaly_results(self, filename='anomaly_results.csv'):
        """
        Exportar resultados de anomalías a CSV
        """
        print(f"\n=== EXPORTANDO RESULTADOS A {filename} ===")
        
        all_anomalies = []
        
        methods = [
            ('Euclidiana', self.anomalies_euclidean),
            ('Mahalanobis', self.anomalies_mahalanobis),
            ('Isolation_Forest', self.anomalies_isolation),
            ('Prophet', self.anomalies_prophet)
        ]
        
        for method_name, anomalies in methods:
            if anomalies is not None:
                # El score para Prophet es el valor real, para los otros es el score de anomalía
                scores = anomalies.values if isinstance(anomalies, pd.Series) else [np.nan] * len(anomalies)
                for date, score in zip(anomalies.index, scores):
                    all_anomalies.append({
                        'fecha': date.strftime('%Y-%m-%d %H:%M:%S'),
                        'metodo': method_name,
                        'score_o_valor': score
                    })
        
        if all_anomalies:
            df_results = pd.DataFrame(all_anomalies)
            df_results.to_csv(filename, index=False)
            print(f"Resultados exportados exitosamente a {filename}")
            print(f"Total de registros: {len(all_anomalies)}")
        else:
            print("No hay anomalías para exportar")
    
    def run_complete_analysis(self, electrical_path, environment_path, irradiance_path):
        """
        Ejecuta el análisis completo siguiendo todos los puntos requeridos
        """
        print("=== INICIANDO ANÁLISIS COMPLETO DE ANOMALÍAS EN SISTEMAS SOLARES ===")
        
        # Puntos 1, 2, 3, 4, 5
        df_combined = self.load_and_merge_datasets(electrical_path, environment_path, irradiance_path)
        
        if df_combined is None: return
        
        df_processed = self.clean_and_preprocess_data(df_combined)
        if df_processed is None: return
        
        self.exploratory_data_analysis(df_combined)
        self.temporal_analysis(df_processed.copy())
        df_engineered = self.feature_engineering(df_processed.copy())
        
        self.analyze_bivariate_relationships(df_engineered)
        # Selección de variables para los métodos multivariados
        df_selected = self.select_variables(df_engineered)
        if df_selected.empty:
            print("Error: No se pudieron seleccionar variables válidas")
            return
        
        print(f"\nDataset final para análisis: {df_selected.shape}")
        
        # Punto 6: Ejecutar detectores de anomalías
        print("\n" + "="*60)
        print("EJECUTANDO DETECTORES DE ANOMALÍAS")
        print("="*60)
        
        # Métodos multivariados
        self.euclidean_distance_detector(df_selected, percentile=95)
        self.mahalanobis_distance_detector(df_selected, percentile=95)
        self.isolation_forest_detector(df_selected, contamination=0.05)

        # Método de series temporales (univariado)
        self.prophet_anomaly_detector(df_engineered, interval_width=0.99)
        
        # Puntos finales: Comparación y reportes
        self.compare_anomaly_methods()
        anomaly_dates = self.generate_anomaly_dates_list()
        self.analyze_anomaly_context()
        self.create_summary_visualization()
        self.export_anomaly_results()
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETO TERMINADO")
        print("="*60)
        
        return {
            'processed_data': df_engineered,
            'anomaly_dates': anomaly_dates,
            'euclidean_anomalies': self.anomalies_euclidean,
            'mahalanobis_anomalies': self.anomalies_mahalanobis,
            'isolation_anomalies': self.anomalies_isolation,
            'prophet_anomalies': self.anomalies_prophet
        }

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del detector
    detector = SolarAnomalyDetector()
    
    # Ejemplaza con las rutas reales de tus archivos
    # Scutar análisis completo
    # Reee necesitan los archivos CSV para ejecutar esta parte.
    results = detector.run_complete_analysis(
        electrical_path='data/electrical_data.csv',
        environment_path='data/environment_data.csv', 
        irradiance_path='data/irradiance_data.csv'
    )
    
    print("\nAnálisis finalizado. Para ejecutar, descomenta el bloque final y proporciona las rutas a los archivos de datos.")