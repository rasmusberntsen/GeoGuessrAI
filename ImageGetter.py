from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import requests
from PIL import Image
def image_getter(area: str):
    num_images = 200
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    countries = [area]
    # Get html from randomstreetview.com
    for country in countries:
        driver.get(f"https://www.randomstreetview.com/{country}")
        # Load images
        im = 0
        while im < num_images:
            # Get html
            html = driver.page_source
            # Find the div address and store it in a variable
            address = driver.find_element(By.XPATH, '//*[@id="address"]').get_attribute('textContent')
            # Find where "google.com/maps" is in the html
            start = html.find("google.com/maps")
            # Find where that string ends
            end = html.find('"', start)
            # Get the url
            url = html[start:end]
            # Remove anything before data
            url = url[url.find("data"):]
            # Remove before !1s
            url = url[url.find("!1s")+3:]
            # Remove after !
            panoid = url[:url.find("!")]
            if panoid == "":
                driver.refresh()
                print("Missing link")
                continue
            images = []
            for x in [0, 1, 2, 3]:
                for y in [0, 1]:
                    url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid={panoid}&output=tile&x={x}&y={y}&zoom=2&nbt=1&fover=2"
                    # Get the image from the url wth Image
                    image = Image.open(requests.get(url, stream=True).raw)
                    # Add the image to the list
                    images.append(image)

            # Stich the images together
            # Create a new image
            new_image = Image.new("RGB", (2048, 1024))
            # Paste the images
            new_image.paste(images[0], (0, 0))
            new_image.paste(images[1], (0, 512))
            new_image.paste(images[2], (512, 0))
            new_image.paste(images[3], (512, 512))
            new_image.paste(images[4], (1024, 0))
            new_image.paste(images[5], (1024, 512))
            new_image.paste(images[6], (1536, 0))
            new_image.paste(images[7], (1536, 512))


            # Save the image as the name Address
            new_image.save(f"images/{country}/{address}.png")
            im += 1
            # Reload the page
            driver.refresh()
            print(f"Saved {address}") 


if __name__ == "__main__":
    from multiprocessing import Process
    p1 = Process(target=image_getter, args=("ee",))
    p1.start()
    p2 = Process(target=image_getter, args=("es",))
    p2.start()
    p3 = Process(target=image_getter, args=("fr",))
    p3.start()
    p4 = Process(target=image_getter, args=("gb",))
    p4.start()
    p5 = Process(target=image_getter, args=("gr",))
    p5.start()
    p6 = Process(target=image_getter, args=("it",))
    p6.start()
    p7 = Process(target=image_getter, args=("pl",))
    p7.start()
    p8 = Process(target=image_getter, args=("ro",))
    p8.start()
    p9 = Process(target=image_getter, args=("ua",))
    p9.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
 