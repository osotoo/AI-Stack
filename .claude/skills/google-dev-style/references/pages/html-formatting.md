## Line length


Break lines at 80 characters except in the following cases:

- Information in a `meta` element at the beginning of a file must be on a single line, so those lines can be as long as needed.
- If a URL in a link has a line break, the link won't work. If a URL is longer than 80 characters (quite common), you're stuck with it. In that case, put the URL on its own line with the `href` attribute to make it easier to review the text before and after, as the following example shows:

```
You can find more information in
<a href="https://example.com/long-url/johan-gambolputty-de-von-ausfern-…-von-hautkopf-of-ulm.html"
>his biography.</a>
```


Break code snippets (in `<pre>` blocks) at 80 characters:

- Older files might use different line lengths. If you're making small changes to a file that has a consistent line length other than 80 characters, then make your changes conform to that file's line length rather than reformatting the whole file.
-  When adding line breaks, make sure that you don't change the meaning of the code! If you're not familiar with the programming language, ask for help from someone who is. But sometimes you just can't avoid a long line.
