import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
# Asegúrate de que python-telegram-bot versión 13.x está en requirements.txt
from telegram import Update 
from telegram.ext import CallbackContext, CommandHandler, Updater

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuración de API-Football 
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY") 
API_URL = "https://v3.football.api-sports.io"

# Placeholder para el modelo TensorFlow
model = None
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)), 
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation='sigmoid') 
    ])
    # model = None # Deja esto
# try:
#     model = tf.keras.Sequential([ ... ]) # Comenta estas líneas
#     model.compile(...)
#     logger.info("Modelo TensorFlow de placeholder CREADO y compilado.")
# except Exception as e:
#     logger.error(f"Error al inicializar el modelo TensorFlow de placeholder: {e}")
#     model = None # Asegúrate que model siga siendo None
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=['accuracy']) 
    logger.info("Modelo TensorFlow de placeholder CREADO y compilado.")
except Exception as e:
    logger.error(f"Error al inicializar el modelo TensorFlow de placeholder: {e}")
    model = None

def get_live_matches():
    if not API_FOOTBALL_KEY:
        logger.error("API_FOOTBALL_KEY no configurada correctamente.")
        return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        response = requests.get(f"{API_URL}/fixtures?live=all", headers=headers, timeout=15)
        response.raise_for_status() 
        data = response.json()
        return data.get('response', [])
    except requests.exceptions.Timeout:
        logger.error("Timeout al obtener partidos de API-Football.")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al obtener partidos de API-Football: {e}")
        return []
    except ValueError as e: 
        logger.error(f"Error al decodificar JSON de API-Football: {e}")
        return []

def preprocess_match_data_placeholder(match_data):
    # Esta función sigue siendo un placeholder por ahora.
    # Solo la llamamos para que el flujo continúe.
    # La reemplazaremos cuando tengamos la estructura de match_info.
    logger.info(f"Usando preprocess_match_data_placeholder para partido: {match_data.get('fixture',{}).get('id','N/A')}")
    return np.random.rand(1, 10).astype(np.float32)


def format_prediction_message(match, probabilities):
    home_team = match.get('teams', {}).get('home', {}).get('name', 'Equipo Local')
    away_team = match.get('teams', {}).get('away', {}).get('name', 'Equipo Visitante')
    
    if not isinstance(probabilities, (list, np.ndarray)) or len(probabilities) != 6:
        logger.error(f"Error: Se esperaban 6 probabilidades, pero se obtuvieron: {probabilities}")
        return f"⚠️ Error procesando predicciones para {home_team} vs {away_team}."

    message = (
        f"⚽ Partido: {home_team} vs {away_team}\n"
        f"⚡ Predicciones con potencial (modelo placeholder, umbral 0.01% para mostrar algo):\n"
    )
    predictions_made = 0
    targets = [
        "8+ Tiros de Esquina Totales", "5+ Tiros de Esquina (1er Tiempo)", "2+ Goles Totales",
        "Gol en los próximos 5 minutos", "3+ Tarjetas Amarillas Totales", "Penal"
    ]

    for i in range(6):
        if probabilities[i] >= 0.0001: # Cambia 0.0001 a 0.85 cuando tengas un modelo real.
            message += f"  - {targets[i]}: {probabilities[i]*100:.1f}%\n"
            predictions_made += 1
    
    if predictions_made == 0:
        return f"➡️ {home_team} vs {away_team}: No hay señales según el modelo placeholder."
    return message

def predict_command(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    logger.info(f"Usuario {user.first_name} ({user.id}) solicitó /predict.")
    
    if model is None:
        update.message.reply_text("⚠️ El modelo de predicción no está disponible. ¡Houston, tenemos un problema! Contacta al administrador.")
        return

    live_matches = get_live_matches()
    if not live_matches:
        update.message.reply_text("📡 No hay partidos en vivo en este momento para analizar, o hubo un error con la API. ¡Intenta más tarde! ⚽")
        return

    if not isinstance(live_matches, list):
        logger.error(f"live_matches no es una lista: {type(live_matches)}. Contenido: {live_matches}")
        update.message.reply_text("Error interno al procesar los datos de los partidos. Por favor, avisa al desarrollador.")
        return

    predictions_output = []
    found_any_match_to_process = False

    for match_info in live_matches:
        if not isinstance(match_info, dict): 
            logger.warning(f"Elemento en live_matches no es un diccionario: {match_info}")
            continue
        
        found_any_match_to_process = True
        fixture_id = match_info.get('fixture', {}).get('id', 'Desconocido')
        logger.info(f"Procesando partido ID: {fixture_id}")
        
        # --- ESTA ES LA LÍNEA NUEVA QUE AÑADIMOS ---
        logger.info(f"DETALLE COMPLETO DEL PARTIDO (match_info): {match_info}") 
        # --- FIN DE LA LÍNEA NUEVA ---
        
        input_data = preprocess_match_data_placeholder(match_info) 

        try:
            pred_probabilities = model.predict(input_data)[0] 
            logger.info(f"Probabilidades crudas (placeholder) para partido ID {fixture_id}: {pred_probabilities}")
        except Exception as e:
            logger.error(f"Error durante model.predict() para partido ID {fixture_id}: {e}")
            predictions_output.append(f"⚠️ Error prediciendo para partido {fixture_id}.")
            continue 

        message_part = format_prediction_message(match_info, pred_probabilities)
        if message_part:
            predictions_output.append(message_part)
    
    if not found_any_match_to_process and live_matches: 
        final_message = "Hubo un problema con los datos de los partidos recibidos. Reintentando más tarde."
    elif predictions_output:
        final_message = "🔮 ¡Predicciones Placeholder! 🔮 (Resultados aleatorios del modelo de prueba)\n\n" + "\n\n---\n\n".join(predictions_output)
        if len(final_message) > 4096: 
            final_message = final_message[:4090] + "\n(...)"
    else:
        final_message = "🤔 No se encontraron partidos o datos para procesar en este momento. ¡El universo dirá más tarde!"

    update.message.reply_text(final_message)

def start_command(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    logger.info(f"Usuario {user.first_name} ({user.id}) inició el bot con /start.")
    update.message.reply_text(
        f'¡Hola, {user.first_name}! 😎 Soy tu bot de predicciones de fútbol desde Colombia. '
        'Usa /predict para obtener predicciones (actualmente de prueba) de partidos en vivo.'
    )

def main_bot_logic():
    logger.info("Iniciando el bot de predicciones (función main_bot_logic)...")
    
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    global API_FOOTBALL_KEY 
    API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")

    if not telegram_token:
        logger.critical("¡TELEGRAM_TOKEN no encontrado en las variables de entorno! El bot no puede iniciar.")
        return 
    
    if not API_FOOTBALL_KEY:
        logger.warning("¡API_FOOTBALL_KEY no encontrado! Las predicciones con datos reales no funcionarán.")
    
    if model is None:
         logger.warning("¡Modelo TensorFlow no cargado/creado! Las predicciones no funcionarán como se espera.")

    updater = Updater(telegram_token, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start_command))
    dispatcher.add_handler(CommandHandler("predict", predict_command))

    logger.info("Empezando el polling del bot...")
    updater.start_polling()
    logger.info("¡Bot listo y escuchando!")
    updater.idle()
    logger.info("Bot detenido.")

if __name__ == '__main__':
    main_bot_logic()