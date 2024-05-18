import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from groq import Groq
import translators as ts
from text_handler import TextHandler
from io import BytesIO
import cv2
import numpy as np
import os
from aiogram.contrib.fsm_storage.memory import MemoryStorage
lang_dict = {"English": "en", "Русский": "ru", "Қазақ": "kk"}
os.environ['GROQ_API_KEY'] = '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'


class Form(StatesGroup):
    choosing_language = State()

class LeafletChatBot:
    def __init__(self, token):
        self.token = token
        self.bot = Bot(token=token)
        self.dp = Dispatcher(self.bot, storage=MemoryStorage())
        self.text_handler = TextHandler(debug=True)
        self.llm_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.chat_history = []
        self.language = 'English'

    async def start_command(self, message: types.Message):
        keyboard_markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        keyboard_markup.add(types.KeyboardButton(text="Қазақ"))
        keyboard_markup.add(types.KeyboardButton(text="Русский"))
        keyboard_markup.add(types.KeyboardButton(text="English"))
        await message.reply(
            "Hello, choose bot's language:", reply_markup=keyboard_markup
        )
        await Form.choosing_language.set()

    async def set_language(self, message: types.Message, state: FSMContext):
        self.language = message.text

        if self.language in ["Қазақ", "Русский", "English"]:
            mess = f"Bot now on {self.language} mode. You can send your document image for text extracting!"
            await message.reply(
                self.translate(mess)
            )
            await state.finish()
        else:
            await message.reply(
                "Invalid language choice. Please choose 'Қазақ' or 'English' or 'Русский'."
            )

    async def handle_image(self, message: types.Message):
        photo = message.photo[-1]
        file_info = await self.bot.get_file(photo.file_id)
        file_path = file_info.file_path
        file = await self.bot.download_file(file_path)
        await message.reply(
            self.translate("Photo received. Processing...")
        )
        file_bytes = BytesIO(file.getvalue()).read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        text = self.text_handler(image).strip().replace('\n', ' ').replace('\x0c', ' ')
        await message.reply(
            self.translate("Photo processed, ask your question!")
        )
        self.chat_history = [{'role':'user', 'content':text + ' fix this text and answer this question shortly and exactly:'}]
        with open('text.txt', 'w') as f:
            f.write(text)

    async def echo_message(self, message: types.Message):
        user_mess = self.translate(message.text, 'English')
        if len(self.chat_history) > 0:
            if self.chat_history[-1]['role'] == 'user':
                self.chat_history[-1]['content'] += '\n' + user_mess
            else:
                self.chat_history.append({
                    'role':'user',
                    'content':user_mess
                })
            completion = self.llm_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=self.chat_history,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        response = self.translate(completion.choices[0].message.content)
        self.chat_history.append({
            'role':'assistant',
            'content':response
        })
        await message.reply(response)

    async def errors_handler(self, update, exception):
        logging.exception(exception)
        await update.message.reply(self.translate("An error occurred!"))

    def translate(self, message, lang = None):
        if lang is None:
            lang = self.language
        return ts.translate_text(
                        message, to_language=lang_dict[lang], translator="google" if lang != 'ru' else 'yandex'
                    )

    def run(self):
        self.dp.register_message_handler(self.start_command, commands=['start'])
        self.dp.register_message_handler(self.set_language, state=Form.choosing_language)
        self.dp.register_message_handler(self.handle_image, content_types=types.ContentType.PHOTO)
        self.dp.register_message_handler(self.echo_message, content_types=types.ContentType.TEXT)
        self.dp.register_errors_handler(self.errors_handler)

        # Start the bot
        executor.start_polling(self.dp, skip_updates=True)


# Replace 'YOUR_BOT_TOKEN' with your bot's API token
if __name__ == "__main__":
    bot = LeafletChatBot("7053665344:AAFTjp0KPFCYBMdTBfJVeXJmO6x8z0soU7k")
    bot.run()
