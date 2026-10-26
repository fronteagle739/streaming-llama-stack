"""Persist tool activity with a chat message; the WebUI renders it as disclosure cards."""
import json
import re

BLOCK = re.compile(r'\n*```stack-tool\n([^\n]+)\n```\n*')


def activity(call, status, result=None):
    function=call.get('function') or {}
    arguments=function.get('arguments', '{}')
    try:arguments=json.loads(arguments)
    except (ValueError, TypeError):pass
    row=dict(id=call['id'],name=function.get('name','Tool call'),status=status,arguments=arguments)
    if result is not None:row['result']=result
    return '\n\n```stack-tool\n'+json.dumps(row,ensure_ascii=False)+'\n```\n\n'


def outcome(message):
    content=message['content']
    if isinstance(content,list):content=next((p['text'] for p in content if p.get('type')=='text'),'{}')
    return json.loads(content)


def without_activity(text):
    def strip(match):
        try:row=json.loads(match[1])
        except ValueError:return match[0]
        return '\n\n' if isinstance(row,dict) and {'id','name','status'}<=row.keys() else match[0]
    return BLOCK.sub(strip,text)
