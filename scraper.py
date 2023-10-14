from selenium import webdriver
from selenium.webdriver.firefox.options import Options as SeleniumOptions
from selenium.webdriver.common.by import By


def setup_driver():
    # starts browser
    browser = webdriver.Firefox(options=configure_driver())
    browser.get('https://www.chess.com/games')
    # print(driver.page_source)
    return browser


def configure_driver():
    options = SeleniumOptions()
    # options.headless = True
    # options.add_argument("--window-size=1920,1200")
    return options


def iterate_players():
    page = 1
    players = driver.find_elements(By.CLASS_NAME, "post-preview-title")     # gets all player profiles
    while len(players) > 0:
        for player in players:
            print(player.text)  # TODO: iterate through the players games
        page += 1
        driver.get("https://www.chess.com/games?page=" + str(page))
        players = driver.find_elements(By.CLASS_NAME, "post-preview-title")


driver = setup_driver()
iterate_players()
# driver.quit()
