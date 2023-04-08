import os

from aiogram import Bot, types
from aiogram.types import ContentTypes
from aiogram.utils import executor
from aiogram.utils.markdown import text
import logging
from aiogram.dispatcher import Dispatcher
import pandas as pd
from predictor import Predictor
from transaction import Transaction

logging.basicConfig(filename='log', format=u'%(filename)s [ LINE:%(lineno)+3s ]#%(levelname)+8s [%(asctime)s]  %(message)s',
                    level=logging.DEBUG)

TOKEN = "5936761300:AAFYwcIjWwEO7pjXOUqgbGVzaaMi2OOng2o"
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

predictor = Predictor()

@dp.message_handler(commands=['start'])
async def process_hi3_command(message: types.Message):
    await message.reply("Загрузите данные")

# @dp.message_handler(commands = 'file')
# async def send_file(message: types.Document):
#     await message.reply_document(open('settings\\settings.py', 'rb'))

@dp.message_handler(content_types=ContentTypes.ANY)
async def unknown_message(message: types.Message):
    if document := message.document:
        await  document.download(
            destination_file=f'{document.file_name}'
        )
        # await  message.reply(f'{document.file_name}')
        data = pd.read_csv(document.file_name)
        # await message.reply(f'{data.head()}')
        global predict
        predict = predictor.predict(data).tolist()

        data['predict'] = predict
        data.to_csv('predict.csv', index=True)
        await message.reply_document(open('predict.csv', 'rb'))
        os.remove('predict.csv')
        keyboard = types.InlineKeyboardMarkup()
        number_ofpage = len(predict) // 10 + 1
        print('number of page:', number_ofpage)
        page_number = 1
        line_number = 0
        for pr in predict:
            print('line number:', line_number )
            if line_number == 10:
                break
            line_number += 1
            temp_fraud = 'Not fraud'
            if pr:
                temp_fraud = 'Fraud'
            key = types.InlineKeyboardButton(temp_fraud, callback_data=temp_fraud)
            keyboard.add(key)
        key = types.InlineKeyboardButton('▶', callback_data='next00' + str(page_number) + '/' + str(number_ofpage))
        key1 = types.InlineKeyboardButton(str(page_number) + '/' + str(number_ofpage), callback_data='777')
        print('predict: ', pr)
        keyboard.row(key1, key)
        await bot.send_message(message.from_user.id, 'predict', reply_markup=keyboard)

        print(1)
        # await message.reply(f'{predict}'
        # )


@dp.callback_query_handler(lambda callback_query: callback_query.data.startswith('next') or callback_query.data.startswith('prev'))
async def process_callback_button1(callback_query):
    c = callback_query.data.find('/')
    num = int(callback_query.data[6:c])
    all_page = callback_query.data[c+1:]

    line_number = 10 * num
    if(callback_query.data[0:4] == 'prev'):
        line_number = 10 * (num - 2)
        num -= 2

    num += 1


    counter = 0
    keyboard = types.InlineKeyboardMarkup()
    for pr in predict:
        if ((counter > line_number - 1) and (counter < (line_number + 10))):

            temp_fraud = 'Not fraud'
            if pr:
                temp_fraud = 'Fraud'
            key = types.InlineKeyboardButton(temp_fraud, callback_data=temp_fraud)
            keyboard.add(key)
        counter += 1
        print(pr)
    key = types.InlineKeyboardButton('▶', callback_data='next' + callback_query.data[4:6] + str(num) + '/' + all_page)
    key1 = types.InlineKeyboardButton(str(num) + '/' + all_page, callback_data='777')
    key2 = types.InlineKeyboardButton('◀', callback_data='prev' + callback_query.data[4:6] + str(num) + '/' + all_page)
    if (line_number == 0):
        keyboard.row(key1, key)
    elif(line_number/10 + 1 == int(all_page)):
        keyboard.row(key2, key1)
    else:
        keyboard.row(key2, key1, key)
    print(all_page)
    await bot.send_message(callback_query.from_user.id, 'Предсказание', reply_markup=keyboard)
    await bot.delete_message(callback_query.from_user.id, callback_query.message.message_id)

if __name__ == '__main__':
   executor.start_polling(dp)