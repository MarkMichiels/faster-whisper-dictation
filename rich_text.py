"""Restricted Markdown rendering. Model HTML is always inert text.

Only this renderer creates tags; no token attributes are ever copied.
"""
from html import escape
import re


def single_line(text: str) -> str:
    """Flatten every Python line separator, also in fallback/status text."""
    return re.sub(r'[ \t]*[\r\n\v\f\x1c-\x1e\x85\u2028\u2029][ \t\r\n\v\f\x1c-\x1e\x85\u2028\u2029]*', ' ', text)


def literal_paragraphs(text: str) -> str:
    """Render local mail parts without interpreting Markdown."""
    return ''.join('<p>' + '<br>'.join(escape(line) for line in part.splitlines()) + '</p>'
                   for part in text.strip().split('\n\n') if part.strip())


def render(markdown: str) -> tuple[str, str]:
    """Return visible plain text and whitelisted HTML from the same token tree.

    Unsupported block structures fail closed. Links lose their destinations;
    images become alt text. Raw HTML is disabled in the actual Markdown parser.
    """
    from markdown_it import MarkdownIt

    tokens = MarkdownIt('commonmark', {'html': False, 'maxNesting': 20}).parse(markdown)

    def inline(children: list) -> tuple[str, str]:
        plain, html = [], []
        for token in children:
            if token.type in ('text', 'code_inline', 'html_inline'):
                plain.append(token.content)
                html.append(escape(token.content))
            elif token.type in ('softbreak', 'hardbreak'):
                plain.append('\n')
                html.append('<br>')
            elif token.type in ('strong_open', 'strong_close'):
                html.append('<strong>' if token.nesting == 1 else '</strong>')
            elif token.type == 'image':
                p, h = inline(token.children or [])
                plain.append(p)
                html.append(h)
            elif token.type in ('link_open', 'link_close', 'em_open', 'em_close'):
                pass
            else:
                raise ValueError('unsupported_inline')
        return ''.join(plain), ''.join(html)

    plain, html = [], []
    tags = {'paragraph': 'p', 'bullet_list': 'ul', 'ordered_list': 'ol', 'list_item': 'li'}
    for token in tokens:
        if token.type == 'inline':
            p, h = inline(token.children or [])
            plain.append(p)
            html.append(h)
        else:
            base, _, edge = token.type.rpartition('_')
            if base not in tags or edge not in ('open', 'close'):
                raise ValueError('unsupported_block')
            html.append(('<' if edge == 'open' else '</') + tags[base] + '>')
            if edge == 'close':
                plain.append('\n')
    visible = ''.join(plain).strip()
    if not visible:
        raise ValueError('empty_render')
    return visible, ''.join(html)
