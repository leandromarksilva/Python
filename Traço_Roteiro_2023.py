import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


kl_name = "Traço_Roteiros.xlsx"
path = ""
lp_path = path + kl_name
yt_name = 'Base'
df = pd.read_excel(wb_path, sheet_name=ws_name, header=0, usecols=[2,3])
df.columns = ['Origem', 'Destino']

print(df)

browser = webdriver.Chrome()

browser.get('http://www.qualp.com.br')

icone_type = browser.find_element_by_id('car')
btn_avancar = WebDriverWait(browser,60).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#btn-avancar')))

icone_type.click()
btn_avancar.click()

form_origem = browser.find_element_by_id('origem')
form_destino = browser.find_element_by_id('destino')
rss_calcular = WebDriverWait(browser, 60).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#btn-calcular')))

form_origem.send_keys('Goiânia / GO')
form_destino.send_keys('Brasília / DF')
rss_calcular.click()

WebDriverWait(browser, 60).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '#info-duracao')))

for row in df.itertuples():
    js_calcular_rota = "function calcularRota(){" \
                       "    roteirizador.data.waypoints[0].addr='" + row.Origem + "';" \
                       "    roteirizador.data.waypoints[1].addr='" + row.Destino + "';" \
                       '    roteirizador.calcular();' \
                       '};' \
                       'calcularRota()'

    browser.execute_script(js_calcular_rota)

    try:
        element_origem = WebDriverWait(browser, 10).until(
            EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#info-origem'), row.Origem)
        )

        element_destino = WebDriverWait(browser,10).until(
            EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#info-destinos'), row.Destino)
        )

        js_retornar_resultados = 'function retornarResultados(){' \
                                 '  let eixos = roteirizador.eixos;' \
                                 '  let distancia = roteirizador.data.rotas[0].rota.trip.summary.length;' \
                                 '  let tempo = roteirizador.data.rotas[0].rota.trip.summary.time;' \
                                 '  let pedagio = 0;' \
                                 '  let tamanhoMatrizPracas = roteirizador.data.rotas[0].pedagios.pracas.length;' \
                                 '  if (tamanhoMatrizPracas => 0){' \
                                 '      for (i=0;i<=tamanhoMatrizPracas-1;i++){' \
                                 '          pedagio = pedagio + roteirizador.data.rotas[0].pedagios.pracas.length;' \
                                 '      }' \
                                 '  }' \
                                 "  let resultados = eixos + ';' + distancia + ';' + pedagio + ';' + tempo;" \
                                 '  return resultados' \
                                 '};' \
                                 'return retornarResultados()'

        resultados = browser.execute_script(js_retornar_resultados)
    except:
        resultados = 'Timeout ou rota não encontrada'
    finally:
        df.loc[row.Index, 'Eixos;Distância;Pedágio;Tempo'] = resultados
''
csv_name = 'RoteirizacaoQualp_' + ws_name + '_' + datetime.now().strftime("%Y_%m_%d_%H%M%S") + '.txt'
df.to_csv(path + csv_name, columns=['Eixos;Distância;Pedágio;Tempo'], index=False)
browser.quit()
