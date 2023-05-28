const hours = new Date().getHours();
let message;

if (hours < 12) message = 'Доброе утро!';
else message = 'Добрый день!';
if (hours > 16) message = 'Добрый вечер!';

messageSpan.innerText = message;

AOS.init(); // aнимации
