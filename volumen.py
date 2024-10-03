import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np

# Función para obtener la posición relativa de la mano
def obtener_posicion_mano(hand_landmarks, frame_shape):
    puntos_mano = []
    altura, ancho, _ = frame_shape
    
    for punto, landmark in enumerate(mp.solutions.hands.HandLandmark):
        posicion_landmark = hand_landmarks.landmark[landmark]
        x, y = int(posicion_landmark.x * ancho), int(posicion_landmark.y * altura)
        puntos_mano.append((x, y, punto))  # Agregar el número del landmark
    
    return puntos_mano

# Función para dibujar los números de los landmarks
def dibujar_numeros(frame, puntos_mano):
    for (x, y, num) in puntos_mano:
        cv2.putText(frame, str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Configuración de Mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Permitir hasta una mano

# Inicializar el controlador de volumen
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Obtener el rango de volumen (normalmente entre -65.25 y 0.0)
vol_min, vol_max = volume.GetVolumeRange()[:2]

# Almacenar la posición anterior del dedo índice
y_indice_anterior = None

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir frame.")
        break

    # Voltear horizontalmente la imagen
    frame = cv2.flip(frame, 1)

    # Convertir el frame a RGB (Mediapipe requiere RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos en el frame
    resultados = hands.process(frame_rgb)

    # Extraer puntos clave de las manos detectadas
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            # Obtener los puntos clave de la mano con sus números
            puntos_mano = obtener_posicion_mano(hand_landmarks, frame.shape)
            
            # Dibujar los puntos clave de la mano en el frame
            for (x, y, num) in puntos_mano:
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Dibujar un círculo en el punto

            # Mostrar el número de cada punto clave
            dibujar_numeros(frame, puntos_mano)

            # Obtener la posición del dedo índice (índice)
            y_indice = puntos_mano[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP][1]

            if y_indice_anterior is not None:
                # Calcular la diferencia en la posición del dedo índice
                diferencia = y_indice_anterior - y_indice

                # Ajustar el volumen en función de la dirección del movimiento
                if diferencia > 5:  # Mover hacia arriba
                    volumen_actual = volume.GetMasterVolumeLevelScalar()
                    nuevo_volumen = min(volumen_actual + 0.05, 1.0)  # Incrementar el volumen
                    volume.SetMasterVolumeLevelScalar(nuevo_volumen, None)
                elif diferencia < -5:  # Mover hacia abajo
                    volumen_actual = volume.GetMasterVolumeLevelScalar()
                    nuevo_volumen = max(volumen_actual - 0.05, 0.0)  # Reducir el volumen
                    volume.SetMasterVolumeLevelScalar(nuevo_volumen, None)

            # Actualizar la posición anterior del dedo índice
            y_indice_anterior = y_indice

            # Mostrar el volumen en la pantalla
            volumen_porcentaje = int(volume.GetMasterVolumeLevelScalar() * 100)
            cv2.putText(frame, f'Vol: {volumen_porcentaje} %', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el frame con los puntos clave de la mano
    cv2.imshow('Control de Volumen con Dedo', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
