"""
System logiki rozmytej do sterowania nawadnianiem ogrodu
Autor: Łukasz Kos, Emilian Murawski

Instalacja:
1. Pobierz i zainstaluj Pythona z tego źródła: https://www.python.org/downloads/
2. Uruchom terminal.
2.1. Zainstaluj bibliotekę numpy, wpisując w terminalu: python3 -m pip install numpy
2.2. Zainstaluj bibliotekę scikit-fuzzy, wpisując w terminalu: python3 -m pip install scikit-fuzzy

Opis:
Ten program jest systemem logiki rozmytej służącym do sterowania nawadnianiem ogrodu. System przyjmuje trzy parametry
wejściowe: wilgotność gleby (soil moisture), temperaturę powietrza (temperature) i promieniowanie słoneczne
(solar radiation),a następnie na podstawie tych parametrów oblicza czas nawadniania (irrigation time).

Parametry wejściowe:
- Wilgotność gleby (soil moisture): Przyjmuje wartości od 0 do 100, gdzie 0 oznacza suchą glebę, a 100 oznacza glebę bardzo mokrą.
- Temperatura powietrza (temperature): Przyjmuje wartości od 0 do 100, gdzie 0 oznacza bardzo niską temperaturę, a 100 oznacza bardzo wysoką temperaturę.
- Promieniowanie słoneczne (solar radiation): Przyjmuje wartości od 0 do 100, gdzie 0 oznacza brak promieniowania słonecznego, a 100 oznacza bardzo intensywne promieniowanie.

Parametr wyjściowy:
- Czas nawadniania (irrigation time): Określa czas w minutach, jaki należy przeznaczyć na nawadnianie ogrodu na podstawie analizy parametrów wejściowych.

Funkcje przynależności:
- Dla każdego z parametrów wejściowych (wilgotność gleby, temperatura powietrza, promieniowanie słoneczne) oraz
parametru wyjściowego (czas nawadniania) zdefiniowane są funkcje przynależności, które określają, jak dane wartości
wpływają na czas nawadniania.

Reguły rozmyte:
- Zdefiniowane są reguły rozmyte, które określają zależności między parametrami wejściowymi a parametrem wyjściowym.
Na przykład, jeśli gleba jest sucha i temperatura jest niska, to czas nawadniania będzie długi.

Użycie:
- Aby skorzystać z tego systemu, należy ustawić wartości parametrów wejściowych
(wilgotność gleby, temperatura, promieniowanie) za pomocą funkcji `irrigation_system.input[]`.
- Następnie można obliczyć czas nawadniania za pomocą funkcji `irrigation_system.compute()`.
- Wynik można odczytać za pomocą `irrigation_system.output[]` i wyświetlić go na ekranie.

Przykład użycia:

irrigation_system.input['soil moisture'] = 5
irrigation_system.input['temperature'] = 15
irrigation_system.input['solar radiation'] = 30
irrigation_system.compute()
print("Czas Nawadniania:", irrigation_system.output['irrigation time'], "minut")

css


Ten przykład ustawia wartości parametrów wejściowych na niską wilgotność gleby, niską temperaturę
i niskie promieniowanie słoneczne, a następnie oblicza i wyświetla czas nawadniania.

"""


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Antecedents (input/sensor) variables for a fuzzy control system.
soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil moisture')
air_temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
solar_radiation = ctrl.Antecedent(np.arange(0, 101, 1), 'solar radiation')

# Consequent (output/control) variable for a fuzzy control system.
duration = ctrl.Consequent(np.arange(0, 13, 1), 'irrigation time')

# Membership function for soil moisture
soil_moisture['dry'] = fuzz.trapmf(soil_moisture.universe, [0, 0, 20, 25])
soil_moisture['optimal'] = fuzz.trimf(soil_moisture.universe, [20, 25, 30])
soil_moisture['wet'] = fuzz.trapmf(soil_moisture.universe, [25, 30, 100, 100])

# Membership function for air temperature
air_temperature['cold'] = fuzz.trapmf(air_temperature.universe, [0, 0, 15, 20])
air_temperature['medium'] = fuzz.trimf(air_temperature.universe, [15, 20, 25])
air_temperature['hot'] = fuzz.trapmf(air_temperature.universe, [20, 25, 100, 100])

# Membership function for solar radiation
solar_radiation['dark'] = fuzz.trapmf(solar_radiation.universe, [0, 0, 30, 40])
solar_radiation['medium'] = fuzz.trapmf(solar_radiation.universe, [30, 40, 60, 70])
solar_radiation['light'] = fuzz.trapmf(solar_radiation.universe, [60, 70, 100, 100])

# Membership function for irrigation duration
duration['zero'] = fuzz.trimf(duration.universe, [0, 0, 3])
duration['very short'] = fuzz.trimf(duration.universe, [0, 3, 6])
duration['short'] = fuzz.trimf(duration.universe, [3, 6, 9])
duration['long'] = fuzz.trimf(duration.universe, [6, 9, 12])
duration['very long'] = fuzz.trimf(duration.universe, [9, 12, 12])

# Fuzzy rules
rule_wet_soil = ctrl.Rule(soil_moisture['wet'], duration['zero'])
rule_optimal_soil_1 = ctrl.Rule(soil_moisture['optimal'] & air_temperature['cold'], duration['short'])
rule_optimal_soil_2 = ctrl.Rule(soil_moisture['optimal'] & air_temperature['medium'] & solar_radiation['dark'],
                                duration['short'])
rule_optimal_soil_3 = ctrl.Rule(soil_moisture['optimal'] & air_temperature['medium'] & solar_radiation['medium'],
                                duration['short'])
rule_optimal_soil_4 = ctrl.Rule(soil_moisture['optimal'] & air_temperature['medium'] & solar_radiation['light'],
                                duration['very short'])
rule_optimal_soil_5 = ctrl.Rule(soil_moisture['optimal'] & air_temperature['hot'] & solar_radiation['dark'],
                                duration['long'])
rule_optimal_soil_6 = ctrl.Rule(soil_moisture['optimal'] & air_temperature['hot'] & solar_radiation['medium'],
                                duration['very short'])
rule_optimal_soil_7 = ctrl.Rule(soil_moisture['optimal'] & air_temperature['hot'] & solar_radiation['light'],
                                duration['zero'])
rule_dry_soil_1 = ctrl.Rule(soil_moisture['dry'] & air_temperature['cold'], duration['very long'])
rule_dry_soil_2 = ctrl.Rule(soil_moisture['dry'] & air_temperature['medium'] & solar_radiation['dark'],
                            duration['long'])
rule_dry_soil_3 = ctrl.Rule(soil_moisture['dry'] & air_temperature['medium'] & solar_radiation['medium'],
                            duration['long'])
rule_dry_soil_4 = ctrl.Rule(soil_moisture['dry'] & air_temperature['medium'] & solar_radiation['light'],
                            duration['short'])
rule_dry_soil_5 = ctrl.Rule(soil_moisture['dry'] & air_temperature['hot'] & solar_radiation['dark'],
                            duration['very long'])
rule_dry_soil_6 = ctrl.Rule(soil_moisture['dry'] & air_temperature['hot'] & solar_radiation['medium'],
                            duration['very short'])
rule_dry_soil_7 = ctrl.Rule(soil_moisture['dry'] & air_temperature['hot'] & solar_radiation['light'], duration['zero'])

# Fuzzy Control System.
system = ctrl.ControlSystem(
    [rule_wet_soil, rule_optimal_soil_1, rule_optimal_soil_2, rule_optimal_soil_3, rule_optimal_soil_4,
     rule_optimal_soil_5, rule_optimal_soil_6, rule_optimal_soil_7, rule_dry_soil_1, rule_dry_soil_2, rule_dry_soil_3,
     rule_dry_soil_4, rule_dry_soil_5, rule_dry_soil_6, rule_dry_soil_7])

# Calculate results from a ControlSystem.
irrigation_system = ctrl.ControlSystemSimulation(system)

# Input values. For every input range is from 0 to 100
irrigation_system.input['soil moisture'] = 5
irrigation_system.input['temperature'] = 15
irrigation_system.input['solar radiation'] = 30

# Oblicz wynik
irrigation_system.compute()

# Wyświetl wynik
print("Czas Nawadniania:", irrigation_system.output['irrigation time'], "minut")
