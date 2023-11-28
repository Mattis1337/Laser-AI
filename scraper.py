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
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", "./games")
    # options.headless = True
    # options.add_argument("--window-size=1920,1200")
    return options


def iterate_players():
    page = 1
    players = driver.find_elements(By.CLASS_NAME, "post-preview-title")     # gets all player profiles
    while len(players) > 0:
        for player in players:
            iterate_games(player)
        page += 1
        driver.get("https://www.chess.com/games?page=" + str(page))
        players = driver.find_elements(By.CLASS_NAME, "post-preview-title")


def iterate_games(player):
    player_name = player.text.title()
    page = 1
    player.click()

    while driver.find_element(By.ID, "master-games-container") is not None:
        download_games()
        page += 1
        driver.get("https://www.chess.com/games/search?p1=" + player_name + "&page=" + str(page))


def download_games():
    driver.find_element(By.ID, "master-games-check-all").click()
    driver.find_element(By.CLASS_NAME, "master-games-download-button").click()


driver = setup_driver()
iterate_players()
# driver.quit()
