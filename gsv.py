from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import time
from webdriver_manager.firefox import GeckoDriverManager
import pandas as pd
# set up options for driver
options = webdriver.FirefoxOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

NUMBER_OF_SCREENSHOTS = 4

def screenshot_canvas():
    '''
    Take a screenshot of the streetview canvas.
    '''

    # find div id xpath 
    address = driver.find_element(By.XPATH, '//*[@id="address"]').get_attribute('textContent')
    #address = driver.find_element(by='id', value='address').text
    # get current time no decimals
    name = f'canvas_{round(time.time())}.png'
    with open(f'pictures/{name}', 'xb') as f:
        canvas = driver.find_element_by_tag_name('canvas')
        f.write(canvas.screenshot_as_png)

    return name, address
    

def rotate_canvas(secs=5):
    '''
    hold down left arrow key for 1 second
    '''
    
    canvas = driver.find_element_by_tag_name('canvas')
    canvas.click()
    ActionChains(driver).key_down(Keys.ARROW_LEFT).pause(secs).key_up(Keys.ARROW_LEFT).perform()


#driver = webdriver.Firefox(GeckoDriverManager().install(), options=options)
driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=options)


#driver = webdriver.Chrome()
driver.get('https://randomstreetview.com/dk#fullscreen')

# let JS etc. load
time.sleep(2)

pictures = []
addresses = []

# Load data
#df = pd.DataFrame(columns=['name', 'address'])  # If no pictures are taken yet
df = pd.read_csv('pictures.csv') # Else load existing data

for _ in range(0, NUMBER_OF_SCREENSHOTS):
    picture, address = screenshot_canvas()
    pictures.append(picture)
    addresses.append(address)
    rotate_canvas(5)

df_new = pd.DataFrame({'picture': pictures, 'address': addresses})
df = pd.concat([df, df_new], ignore_index=True)
df.to_csv('pictures.csv', index=False)

driver.close()



