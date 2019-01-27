---
title: hexo 功能完善(置顶 评论等)
date: 2019-01-27 09:37:37
tags: hexo
categories: hexo
---

## 置顶

修改Hexo文件夹下`node_modules/hexo-generator-index/lib/generator.js`，添加如下代码

```js
posts.data = posts.data.sort(function(first, second) {
        if (first.top && second.top) { // 两篇文章top都有定义
            return first.top == second.top ? second.date - first.date : second.top - first.top //若top值一样则按照文章日期降序排, 否则按照top值降序排
        } else if (first.top && !second.top) { // 以下是只有一篇文章top有定义，将有top的排在前面
            return -1;
        } else if (!first.top && second.top) {
            return 1;
        } else {
            return second.date - first.date;  // 都没定义top，按照文章日期降序排
        }
    });
```

添加后完整代码如下

```js
'use strict';

var pagination = require('hexo-pagination');

module.exports = function(locals) {
  var config = this.config;
  var posts = locals.posts.sort(config.index_generator.order_by);
  posts.data = posts.data.sort(function(first, second) {
        if (first.top && second.top) { // 两篇文章top都有定义
            return first.top == second.top ? second.date - first.date : second.top - first.top //若top值一样则按照文章日期降序排, 否则按照top值降序排
        } else if (first.top && !second.top) { // 以下是只有一篇文章top有定义，将有top的排在前面
            return -1;
        } else if (!first.top && second.top) {
            return 1;
        } else {
            return second.date - first.date;  // 都没定义top，按照文章日期降序排
        }
    });
  var paginationDir = config.pagination_dir || 'page';
  var path = config.index_generator.path || '';

  return pagination(path, posts, {
    perPage: config.index_generator.per_page,
    layout: ['index', 'archive'],
    format: paginationDir + '/%d/',
    data: {
      __index: true
    }
  });
};
```

在需要置顶的文章front-matter中添加top值（top值越大文章越靠前）

```markdown
title: hexo 功能完善(置顶 评论等)
date: 2019-01-27 09:37:37
tags: hexo
categories: hexo
top: 2
```

## 添加阅读量及访问量统计



### Next升级后添加访问量

升级到v6后只需要修改`next/_config.yml`，则可以在底部显示访问量

```
busuanzi_count:
  enable: true
  total_visitors: true
  total_visitors_icon: user
  total_views: true
  total_views_icon: eye
  post_views: false  # 由于已经使用leancloud_visitors进行了统计，这里置为false
  post_views_icon: eye
```

### 旧版本添加访问量

参考：[Hexo博客Next主题添加文章阅读量及网站访问信息](http://www.mdslq.cn/archives/d93ac7d.html)

写的很详细，需要注意的是最后“显示统计标签”，修改`next/layout/_partials/footer.swig`文件，在该文件开头需要加上

```js
<script async src="https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js">
</script>
```

#### 访问量无法显示

之前如果配置过访问量但现在已无法显示的原因是不蒜子换域名了，导致之前配置的js文件不能正常调用，所以就无法显示。

修改上方文件，将`https://dn-lbstatics.qbox.me/busuanzi/2.3/busuanzi.pure.mini.js`修改为`https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js`，就可以显示了。

另外，使用`hexo s`部署在本地预览效果的时候，uv数和pv数会过大，这是由于不蒜子用户使用一个存储空间，所以使用`localhost:4000`进行本地预览的时候会导致数字异常，这是正常现象，只需要将博客部署至云端即可恢复正常。

## Next主题升级

- 将旧`hexo/themes/next`备份，然后删除
- 克隆新版本的next`git clone https://github.com/theme-next/hexo-theme-next themes/next`
- 参照旧版本设定相关配置，完毕后删除旧版本备份。

显示中文需修改 `hexo/_config.yml`：由`language: zh-Hans`修改为`language: zh-CN`

## 添加评论

Next 6 已经集成这个功能了，可以使用和访问量同一个应用。

1. 在云端的 leancloud 应用中创建一个名为 `Comment` 的类，使用默认的 ACL 权限设置。
2. 在主题配置文件中设置 app_id 和 app_key 即可。

```yaml
valine:
  enable: true # When enable is set to be true, leancloud_visitors is recommended to be closed for the re-initialization problem within different leancloud adk version.
  appid: # your leancloud application appid
  appkey: # your leancloud application appkey
  notify: false # mail notifier, See: https://github.com/xCss/Valine/wiki
  verify: false # Verification code
  placeholder: 在此处输入评论 # comment box placeholder
  avatar: mm # gravatar style
  guest_info: nick,mail,link # custom comment header
  pageSize: 10 # pagination size
  visitor: false # leancloud-counter-security is not supported for now. When visitor is set to be true, appid and appkey are recommended to be the same as leancloud_visitors' for counter compatibility. Article reading statistic https://valine.js.org/visitor.html
  comment_count: true # if false, comment count will only be displayed in post page, not in home page
```

## 添加搜索

安装插件：在hexo根目录下运行`npm install hexo-generator-searchdb --save`

修改`next/_config.yml`

```yaml
# Local search
# Dependencies: https://github.com/theme-next/hexo-generator-searchdb
local_search:
  enable: true
  # if auto, trigger search by changing input
  # if manual, trigger search by pressing enter key or search button
  trigger: auto
  # show top n results per article, show all results by setting to -1
  top_n_per_article: 1
  # unescape html strings to the readable one
  unescape: false
```

## 修改字体及高亮

这部分可参考：[Hexo Next博客优化](https://maoao530.github.io/2017/01/25/hexo-blog-seo/)

### 行间距及行内代码字体颜色调整

由于使用的是Mist模式，所以需要修改：`hexo/themes/next/source/css/_schemes/Mist/_base.styl`

```css
// Tags
// --------------------------------------------------

.posts-expand .post-body ul li {
    list-style: disc;
}

code {
    padding: 2px 4px;
    word-wrap: break-word;
    color: rgba(244, 67, 54, 0.66);
    background: rgba(238, 238, 238, 0.5);
    border-radius: 3px;
    font-size: 14px;
}

h1, h2, h3, h4, h5, h6 { margin: 40px 0 10px; }

h2 {
    font-size: 32px;
}
h3 {
    font-size: 24px;
}

blockquote {
    border-left: 4px solid #42b983;
}

p { margin: 0 0 25px 0; }

a { border-bottom-color: $grey-light; }

hr {
  margin: 20px 0;
  height: 2px;
}
```

### 修改链接颜色

修改`next/source/css/_common/components/post/post.styl`，添加以下代码

```css
.post-body p a{
  color: #0593d3;
  border-bottom: none;

  &:hover {
    color: #ea6753;
  }
}
```

### 修改高亮颜色

默认的高亮颜色有些暗，可以修改`hexo/themes/next/source/css/_common/components/highlight/theme.styl`

因为使用的高亮是"normal"，所以修改这一部分

```css
if $highlight_theme == "normal"
  $highlight-background   = #f8f9fa
  $highlight-current-line = #efefef
  $highlight-selection    = #d6d6d6
  $highlight-foreground   = #4d4d4c
  $highlight-comment      = #8e908c
  $highlight-red          = #c82829
  $highlight-orange       = #f5871f
  $highlight-yellow       = #eab700
  $highlight-green        = #008000
  $highlight-aqua         = #3e999f
  $highlight-blue         = #4271ae
  $highlight-purple       = #AA22FF
  $highlight-gutter       = {
    color: #869194,
    bg-color: #eff2f3
  }
```

